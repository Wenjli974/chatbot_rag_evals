import sqlite3

from trulens_eval import Tru

def create_database(db_name):
    conn = sqlite3.connect(db_name)
    print(f"Database {db_name} created successfully.")

def connect_to_database(db_name):
    conn = sqlite3.connect(db_name)
    return conn

def list_tables(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print("Tables in the database:")
    for table_name in cursor.fetchall():
        print(table_name[0])
    cursor.close()


def create_table(conn, create_table_sql):
    cursor = conn.cursor()
    cursor.execute(create_table_sql)
    conn.commit()
    print("Table created successfully")
    cursor.close()

def insert_data(conn, insert_data_sql):
    cursor = conn.cursor()
    cursor.execute(insert_data_sql)
    conn.commit()
    print("Data inserted successfully")
    cursor.close()


def main():
    conn = connect_to_database('rag.db')
    

    cursor = conn.cursor()
    list_tables(conn)
    print("Data in the table:test")
    cursor.execute("Select * from records")
    rows = cursor.fetchall()
    for row in rows:
        print(row.decode('utf-8'))
    conn.close()

# apps
# feedback_defs
# feedbacks
# records

main()


# import json


# text = [{'args': {'prompt': '2024年4月29号能申请wfh吗？', 'response': '要确定2024年4月29号是否能申请居家办公，首先需要查看该日期是周几，以及那一周的工作日数量。根据政策，如果一周的工作日少于等于3天，则不可以申请居家办公。如果4月29号所在的周工作日超过3天，且员工那周还未申请超过2天的居家办公，那么是可以申请的。'}, 'ret': 0.8, 'meta': {'reason': 'Criteria: The response provides relevant information about the eligibility to apply for working from home on April 29, 2024, based on the day of the week and the number of working days in that week.\nSupporting Evidence: The response explains that to determine if one can apply for working from home on April 29, 2024, it is necessary to check the day of the week and the total number of working days in that week. It further states the policy that if the working days in a week are equal to or less than 3, then working from home cannot be applied for. Additionally, it mentions that if April 29 falls on a week with more than 3 working days and the employee has not already applied for more than 2 days of working from home that week, then it is possible to apply for working from home on that day. This information directly addresses the eligibility criteria for applying for working from home on the specified date, making the response highly relevant.'}}]



# def pretty_print_json(json_dict):
#     pretty_json = json.dumps(json_dict, indent=4, sort_keys=True,ensure_ascii=False)
#     print(pretty_json)

# for item in text:
#     pretty_print_json(item)