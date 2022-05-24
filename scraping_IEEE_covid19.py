import os
from multiprocessing import Pool

import requests
from bs4 import BeautifulSoup
from selenium.webdriver import Firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

file_name = os.path.join(os.getcwd(), "IEEE_covid_tweets.csv")

driver = Firefox()

driver.get(
    'https://ieee-dataport.org/open-access/coronavirus-covid-19-geo-tagged-tweets-dataset')
driver.implicitly_wait(2)  # seconds

driver.find_element_by_link_text('Login').click()
driver.implicitly_wait(5)  # seconds

driver.find_element_by_id('username').send_keys('ali.osama@fci.helwan.edu.eg')

driver.find_element_by_id('password').send_keys('aFLLpXbkDs4Q7c8')

driver.find_element_by_id('modalWindowRegisterSignInBtn').click()
driver.implicitly_wait(5)  # seconds

WebDriverWait(driver, 30).until(
    EC.presence_of_element_located(
        (By.CLASS_NAME, "field-name-field-open-access-files"))
)
soup = BeautifulSoup(driver.page_source, "lxml")
produsts = soup.find(class_='field field-name-field-open-access-files field-type-file field-label-hidden').find(
    class_='list-group list-group-flush')

WebDriverWait(driver, 30).until(
    EC.presence_of_element_located((By.CLASS_NAME, "file"))
)

files_bs_elements = produsts.findAll(class_='file')
download_links = []
file_names = []

for product in files_bs_elements:
    file_names.append(product.get_text())
    download_links.append(product.find('a')['href'])


def _worker_download_file(link):
    try:
        downloaded_obj = requests.get(link)
        with open(file_name, "ab") as file:
            file.write(downloaded_obj.content)
    except:
        pass


threads_num = 10
pool = Pool(threads_num)
pool.map(_worker_download_file, download_links)
pool.close()


driver.close()
