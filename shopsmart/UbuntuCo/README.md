# ShopSmart accounts and checkout

Run the app from this directory with `streamlit run app.py`.

Customers can browse and add products to their cart as guests. Checkout routes
guests to the sign-up/login page and resumes checkout after successful
authentication. Accounts are stored in the local `shopsmart.sqlite3` database.
Passwords are stored as salted PBKDF2 hashes. This SQLite setup is intended
for a single-app demo; use a managed identity service and durable database
before deploying a multi-instance production store.

## PayPal Sandbox setup

1. Sign in to the [PayPal Developer Dashboard](https://developer.paypal.com/).
2. Open **Apps & Credentials**, select **Sandbox**, and create a REST app.
   Copy its **Client ID** and **Secret**. Use Sandbox credentials, not Live
   credentials.
3. From this directory, copy
   `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`. On Windows:

   ```powershell
   Copy-Item .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```

4. Put the Sandbox Client ID and Secret into the matching fields in
   `.streamlit/secrets.toml`. Keep this file private; it is excluded from Git.
5. Run the app with `streamlit run app.py`, then test checkout with a Sandbox
   personal/buyer account from **Sandbox > Accounts** in the PayPal Developer
   Dashboard. Approve the payment on PayPal to return to ShopSmart.

The example uses `http://localhost:8501/` as the local return URL and
`http://localhost:8501/?paypal=cancelled` as the local cancel URL. When hosted,
replace both with the app's public HTTPS URL (including the cancel query
parameter). PayPal checkout requires a browser-accessible URL.

Checkout redirects customers to PayPal. The app will not claim a checkout
succeeded unless PayPal confirms the order capture. ShopSmart does not collect
or store card numbers or security codes; customers add/select their payment
method on PayPal's hosted checkout page.
