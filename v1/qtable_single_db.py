import sqlite3
import os

from v1.position import Position
from v1.qtable import QTable


class SingleQTableDB(QTable):
    def __init__(self, database_name="qtable.db", default_q_value=0):
        self.qtable = {}
        self.database_name = database_name
        self.conn = None
        self.cursor = None
        self.init_db()
        super().__init__(default_q_value)

    @property
    def qtable_to_read(self):
        return self.qtable

    @property
    def qtable_to_write(self):
        return self.qtable

    def init_db(self):
        if os.path.exists(self.database_name):
            os.remove(self.database_name)
        self.conn = sqlite3.connect(self.database_name)
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE qtable (id INTEGER PRIMARY KEY, state TEXT, action TEXT, qvalue REAL)''')
        cursor.execute('''CREATE INDEX qtable_state ON qtable (state)''')
        cursor.execute('''CREATE INDEX qtable_state_action ON qtable (state, action)''')
        cursor.execute('''CREATE INDEX qtable_state_action_qvalue ON qtable (state, action, qvalue)''')
        cursor.execute('''CREATE TABLE qtable_history (id INTEGER PRIMARY KEY, state TEXT, 
                        action TEXT, qvalue REAL, change_time TEXT)''')
        cursor.execute('''CREATE TRIGGER update_qvalue_trg
                        AFTER UPDATE OF qvalue ON qtable
                        FOR EACH ROW
                        BEGIN
                            INSERT INTO qtable_history (state, action, qvalue, change_time)
                            VALUES (OLD.state, OLD.state, NEW.qvalue, datetime('now'));
                        END;''')
        self.conn.close()

    def start(self):
        self.conn = sqlite3.connect(self.database_name)
        self.cursor = self.conn.cursor()

    def finalize(self):
        self.conn.close()

    def select_q_value_db(self, state, action):
        # cursor = self.conn.cursor()
        self.cursor.execute('SELECT qvalue FROM qtable WHERE state = ? AND action = ?', (state, action.__hash__()))
        data = self.cursor.fetchone()
        if data:
            return data[0]
        return self.default_q_value

    def upsert_q_value_db(self, state, action, qvalue):
        # cursor = self.conn.cursor()
        self.cursor.execute('SELECT qvalue FROM qtable WHERE state = ? AND action = ?', (state, action.__hash__()))
        data = self.cursor.fetchone()
        if data:
            self.cursor.execute('UPDATE qtable SET qvalue = ? WHERE state = ? AND action = ?',
                           (qvalue, state, action.__hash__()))
        else:
            self.cursor.execute('INSERT INTO qtable (state, action, qvalue) VALUES (?, ?, ?)',
                           (state, action.__hash__(), qvalue))
        self.conn.commit()

    def select_best_action_db(self, state):
        # cursor = self.conn.cursor()
        self.cursor.execute('SELECT action FROM qtable WHERE state = ? ORDER BY qvalue DESC LIMIT 1', (state,))
        data = self.cursor.fetchone()
        if data[0]:
            return Position.from_hash(data[0])
        else:
            return None

    def select_max_q_value_db(self, state):
        # cursor = self.conn.cursor()
        self.cursor.execute('SELECT MAX(qvalue) AS qvalue FROM qtable WHERE state = ?', (state,))
        data = self.cursor.fetchone()
        if data[0]:
            return data[0]
        return self.default_q_value

    def select_states_count_db(self):
        # cursor = self.conn.cursor()
        self.cursor.execute('SELECT COUNT(DISTINCT state) AS total_states FROM qtable')
        data = self.cursor.fetchone()
        return data[0]

    def select_all_states_db(self):
        # cursor = self.conn.cursor()
        self.cursor.execute('SELECT DISTINCT state FROM qtable')
        return self.cursor.fetchall()

    def get_q_value(self, state, action):
        return self.select_q_value_db(state, action)

    def update_q_value(self, state, action, value):
        self.upsert_q_value_db(state, action, value)

    def get_best_action(self, state):
        return self.select_best_action_db(state)

    def get_max_q_value(self, state):
        return self.select_max_q_value_db(state)

    def qtable_size(self):
        return self.select_states_count_db()

    def all_states(self):
        return self.select_all_states_db()
