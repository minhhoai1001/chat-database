import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase

load_dotenv()

class MySQLClient():
    def __init__(self, host, port, password):
        encoded_password = quote_plus(password)
        mysql_uri = f"mysql+mysqlconnector://root:{encoded_password}@{host}:{port}/client_1"
        self.db = SQLDatabase.from_uri(mysql_uri)
        self.schema = self.db.get_table_info()

    def get_table_info(self):
        return self.schema

    def run_query(self, query):
        return self.db.run(query)



if __name__ == "__main__":
    MYSQL_SERVER = os.environ['MYSQL_SERVER']
    MYSQL_PORT = os.environ['MYSQL_PORT']
    MYSQL_PASSWORD = os.environ['MYSQL_PASSWORD']
    
    mysql_client = MySQLClient(MYSQL_SERVER, MYSQL_PORT, MYSQL_PASSWORD)
    # print(mysql_client.get_table_info())
    retries = mysql_client.run_query("SELECT * FROM project_export_history LIMIT 2")
    print(retries)
