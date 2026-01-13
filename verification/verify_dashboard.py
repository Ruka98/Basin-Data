from playwright.sync_api import Page, expect, sync_playwright
import time

def verify_dashboard(page: Page):
    # 1. Arrange: Go to the dashboard
    page.goto("http://localhost:7860")

    # 2. Act: Navigate to WA+ Analysis tab
    page.get_by_text("WA+ Analysis").click()

    # Wait for the tab to load
    page.wait_for_selector("#basin-dropdown")

    # Verify the layout
    # Check if "Select Basin" label is present
    expect(page.get_by_text("Select Basin")).to_be_visible()

    # Take a screenshot of the initial state (Global View)
    time.sleep(2) # Wait for map to render
    page.screenshot(path="verification/dashboard_initial.png")

    # Select a basin
    page.locator("#basin-dropdown").click()
    page.get_by_text("Amman Zarqa").click()

    # Wait for update
    time.sleep(5) # Wait for callbacks (map zoom, year selection reveal)

    # Verify Year selection appeared
    expect(page.locator("#year-selection-panel")).to_be_visible()

    # Take a screenshot of the selected state
    page.screenshot(path="verification/dashboard_selected.png")

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": 1600, "height": 1000}) # Large desktop
        try:
            verify_dashboard(page)
        finally:
            browser.close()
