import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from recommender import Recommender


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rec = Recommender("artifacts.joblib")  # loaded once at startup
    yield


app = FastAPI(title="ShopSmart Recommender", lifespan=lifespan)

origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")  # set to your site's domain in production
app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_methods=["GET", "POST"], allow_headers=["*"])


class CartItem(BaseModel):
    item: str = Field(min_length=1, description="StockCode or product description")
    quantity: int = Field(default=1, ge=1)


class CartRequest(BaseModel):
    cart: list[CartItem] = Field(min_length=1)
    top_n: int = Field(default=3, ge=1, le=5)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/types")
def list_types(request: Request):
    return {"types": request.app.state.rec.list_types()}


@app.post("/recommend")
def recommend(req: CartRequest, request: Request):
    try:
        recs = request.app.state.rec.recommend(
            [(c.item, c.quantity) for c in req.cart], req.top_n)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"recommendations": recs}


@app.post("/score")
def score(req: CartRequest, request: Request):
    try:
        scores = request.app.state.rec.score_all(
            [(c.item, c.quantity) for c in req.cart])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"scores": scores}