import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driverPath = "/home/hieubigby/Applications/ChromeDriver/chromedriver"
driver = webdriver.Chrome()

driver.get('https://teachablemachine.withgoogle.com/train/image')
print(driver.title)

# link = driver.find_element(By.LINK_TEXT, 'Tutorials')
# link.click()

time.sleep(5)

# el = WebDriverWait(driver, timeout=10).until(lambda d: d.find_element(By.ID, 'onboard-box'))
element = driver.find_element(By.ID, 'onboard-box')
print(element.text)
# element.send_keys('test')

# try:
#     element = WebDriverWait(driver, 10).until(
#         EC.presence_of_element_located((By.ID, 'label-input'))
#     )
#     element.click()
#     element = driver.find_element(By.ID, 'label-input')
#     element.send_keys('test')
#
#     # element = WebDriverWait(driver, 10).until(
#     #     EC.presence_of_element_located((By.CSS_SELECTOR, '.button__ButtonStyled-sc-73bwj8-0.hZeIKp'))
#     # )
#     # element.click()
#
# finally:
#     # driver.quit()
#     print('Ending section...')
#     while(True):
#         pass

while(True):
    pass