# ShopSmart accounts and checkout

Run the app from this directory with `streamlit run app.py`.

Customers can browse and add products to their cart as guests. Checkout routes
guests to the sign-up/login page and resumes checkout after successful
authentication. Accounts are stored in the local `shopsmart.sqlite3` database.
Passwords are stored as salted PBKDF2 hashes. This SQLite setup is intended
for a single-app demo; use a managed identity service and durable database
before deploying a multi-instance production store.

Checkout redirects customers to PayPal. Copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and replace the
Sandbox placeholders with PayPal Sandbox app credentials. Set the return and
cancel URLs to this app's externally reachable URL when deployed. Keep the
secrets file private. The app will not claim a checkout succeeded unless
PayPal confirms the order capture. ShopSmart does not collect or store card
numbers or security codes; customers add/select their payment method on
PayPal's hosted checkout page.
