from playwright.sync_api import sync_playwright

URL = "https://made-to-give.lovable.app/"
NUM_VISITS = 2       # Change the number of sessions here
VIEW_TIME_MS = 3000

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)

    try:
        for visit in range(1, NUM_VISITS + 1):
            context = browser.new_context()

            try:
                page = context.new_page()
                page.goto(URL, wait_until="load")
                page.wait_for_timeout(VIEW_TIME_MS)

                page.get_by_role(
                    "link",
                    name="Browse one-off cake sessions",
                    exact=True,
                ).click()

                page.wait_for_url("**/sessions?cta=primary")
                page.wait_for_timeout(VIEW_TIME_MS)

                print(f"Session {visit}/{NUM_VISITS}: reached {page.url}")

            except Exception as error:
                print(f"Session {visit}/{NUM_VISITS} failed: {error}")

            finally:
                context.close()

    finally:
        browser.close()