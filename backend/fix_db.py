import sqlite3

conn = sqlite3.connect('ContextAI.db')
cursor = conn.cursor()

try:
    cursor.execute("DROP TABLE users;")
    print("Dropped users table")
except Exception as e:
    print(f"Error dropping users table: {e}")

try:
    cursor.execute("ALTER TABLE workspaces DROP COLUMN user_id;")
    print("Dropped user_id from workspaces")
except Exception as e:
    print(f"Error dropping user_id: {e}")
    
conn.commit()
conn.close()
