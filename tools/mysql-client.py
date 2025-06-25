import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase

load_dotenv()

MYSQL_SERVER = os.environ['MYSQL_SERVER']
MYSQL_PORT = os.environ['MYSQL_PORT']
MYSQL_PASSWORD = os.environ['MYSQL_PASSWORD']

encoded_password = quote_plus(MYSQL_PASSWORD)
mysql_uri = f"mysql+mysqlconnector://root:{encoded_password}@{MYSQL_SERVER}:{MYSQL_PORT}/client_1"

db = SQLDatabase.from_uri(mysql_uri)
schema = db.get_table_info()
print(schema)
