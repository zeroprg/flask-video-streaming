import sqlite3
from sqlite3 import Error
 
 
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
 
    return conn
 
 
def select_all_stats(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT hashcode, currentdate, currentime, type FROM objects")
 
    rows = cur.fetchall()
 
    #for row in rows:
    #    print(row)
    return rows

def insert_frame(conn, hashcode, date, time, type , numpy_array, x_dim, y_dim):
    cur = conn.cursor()
    cur.execute("INSERT INTO objects (hashcode, currentdate, currentime, type, frame, x_dim, y_dim) VALUES (?, ?, ?, ?, ?, ?, ?)", (hashcode, date, time, type, numpy_array.tobytes(), x_dim, y_dim))
    conn.commit()

def select_frame_by_time(conn, time1, time2):
    """
    Query frames by time
    :param conn: the Connection object
    :param time1, time2 in TEXT format  as ISO8601 strings ("YYYY-MM-DD HH:MM:SS.SSS")
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM objects WHERE currentime BETWEEN ? and ?", (time1,time2,))
 
    rows = cur.fetchall()
 
    #for row in rows:
    #    print(row)
    return rows
 
 
def main():
    database = "framedata.db"
 
    # create a database connection
    conn = create_connection(database)
    with conn:
        print("1. Query objects by time:")
        select_frame_by_time(conn, "2019-01-01 00:00:00.00.000", "2019-12-31 00:00:00.00.000")
 
        print("2. Query all objects")
        select_all_stats(conn)
 
 
if __name__ == '__main__':
    main()
