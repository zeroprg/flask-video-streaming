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

def insert_statistic(conn, params):
    cur = conn.cursor()
    for param in params:
       try:
          cur.execute("INSERT INTO statistic(type,currentime,y,text,cam) VALUES (?, ?, ?, ?, ?)", (param.name, param.x, param.y, param.text, param.cam))
       except Error as e:
          print("Error during insertion of statistic {}".format(e))
    conn.commit()
    


def insert_frame(conn, hashcode, date, time, type , numpy_array, x_dim, y_dim, cam):
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO objects (hashcode, currentdate, currentime, type, frame, x_dim, y_dim,cam) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (str(hashcode), date, time, type, numpy_array.tobytes(), x_dim, y_dim, cam))
    except:
        print("Hashcode : {} already existed for object {} and cam: {}".format(hashcode, type, cam))
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
def delete_frames_later_then(conn, predicate):
    """
    Delete all records from objects table which are later then 'predicate'
    predicate : '-70 minutes' , '-1 seconds '
    """
    cur = conn.cursor()
    cur.execute("DELETE from objects WHERE currentime < datetime('now'," + predicate+ ")")
 
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
