import time
import random
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc

URL = "http://127.0.0.1:5000/"

def get_browser(user_agent):
    options = uc.ChromeOptions()
    options.add_argument(f"user-agent={user_agent}")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    
    # Make Chrome visible
    # (DON'T use headless)
    
    driver = uc.Chrome(options=options)
    return driver

def get_user_agent():
    # Just return a random common user agent
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
    ]
    return random.choice(user_agents)

def visit_site():
    user_agent = get_user_agent()
    print(f"[+] Starting Chrome with user agent: {user_agent}")
    driver = get_browser(user_agent)

    try:
        print(f"[+] Visiting {URL}")
        driver.get(URL)

        print("[✓] Page loaded. Waiting for form...")

        # Wait until the form fields are available
        wait = WebDriverWait(driver, 10)
        name_input = wait.until(EC.presence_of_element_located((By.ID, "name")))
        email_input = wait.until(EC.presence_of_element_located((By.ID, "email")))

        # Simulate typing
        name = f"TestUser{random.randint(100, 999)}"
        email = f"user{random.randint(1000,9999)}@example.com"
        print(f"[→] Filling name: {name}")
        name_input.send_keys(name)
        time.sleep(random.uniform(0.5, 1))

        print(f"[→] Filling email: {email}")
        email_input.send_keys(email)
        time.sleep(random.uniform(0.5, 1))

        # Submit the form
        submit_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']")))
        print("[→] Submitting form...")
        submit_button.click()

        print("[✓] Form submitted successfully!")

        time.sleep(3)
        driver.quit()

    except Exception as e:
        print(f"[!] Error: {e}")
        driver.quit()

if __name__ == "__main__":
    visit_site()
