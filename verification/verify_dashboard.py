from playwright.sync_api import sync_playwright, expect

def verify_dashboard():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Navigate to the app
        print("Navigating to app...")
        page.goto("http://localhost:7860")

        # 2. Click "WA+ Analysis" tab
        print("Clicking WA+ Analysis tab...")
        page.get_by_text("WA+ Analysis").click()

        # 3. Select "Amman Zarqa" from dropdown
        print("Selecting Amman Zarqa...")
        # Using specific selector for the dropdown input
        page.locator("#basin-dropdown").click()
        page.get_by_text("Amman Zarqa", exact=True).click()

        # 4. Wait for content to load
        print("Waiting for content...")
        # Wait for the map graph to appear
        expect(page.locator("#lu-map-graph")).to_be_visible(timeout=30000)

        # 5. Verify Structure
        print("Verifying structure...")

        # Year Selection should be visible
        expect(page.get_by_text("Select Year Range")).to_be_visible()

        # Land Use Map should be visible
        expect(page.locator("#lu-map-graph")).to_be_visible()

        # Land Use Table should be visible (check for a header)
        # Dash Table headers might not be strictly "columnheader" role in all versions or configs,
        # or it might be shadowed. Using class selector to be safe.
        expect(page.locator(".dash-spreadsheet-container")).to_be_visible()
        expect(page.get_by_text("Water Management Class", exact=True)).to_be_visible()

        # Site Description (Study Area) should be visible
        expect(page.get_by_text("Amman Zarqa Study Area")).to_be_visible()

        # 6. Take Screenshot
        print("Taking screenshot...")
        page.screenshot(path="/home/jules/verification/dashboard_analysis.png", full_page=True)

        browser.close()

if __name__ == "__main__":
    verify_dashboard()
