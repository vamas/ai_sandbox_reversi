from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, func, select, insert, update, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import NoResultFound
from datetime import datetime
import os
from v1.position import Position
from v1.qtable import QTable

Base = declarative_base()



class QTableModel(Base):
    __tablename__ = "qtable"
    id = Column(Integer, primary_key=True, autoincrement=True)
    state = Column(String, index=True, nullable=False)
    action = Column(String, index=True, nullable=False)
    qvalue = Column(Float, index=True,  nullable=False)
    __table_args__ = (
        Index('idx_state_action', 'state', 'action'),
    )

class QTableHistory(Base):
    __tablename__ = "qtable_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    state = Column(String, nullable=False)
    action = Column(String, nullable=False)
    qvalue = Column(Float, nullable=False)
    change_time = Column(DateTime, default=datetime.utcnow)


class SingleQTableDBPlus(QTable):
    def __init__(self, database_name, default_q_value=0):
        self.database_name = "sqlite:///" + database_name
        self.default_q_value = default_q_value
        self.engine = create_engine(self.database_name, echo=False)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        super().__init__(default_q_value)

    def start(self):
        self.session = self.Session()

    def finalize(self):
        self.session.close()

    @property
    def qtable_to_read(self):
        pass

    @property
    def qtable_to_write(self):
        pass

    def select_q_value_db(self, state, action):
        try:
            qvalue = self.session.execute(
                select(QTableModel.qvalue).where(
                    QTableModel.state == state, QTableModel.action == str(action.__hash__())
                )
            ).scalar_one()
            return qvalue
        except NoResultFound:
            return self.default_q_value

    def upsert_q_value_db(self, state, action, qvalue):
        action_hash = str(action.__hash__())
        existing = self.session.execute(
            select(QTableModel).where(
                QTableModel.state == state, QTableModel.action == action_hash
            )
        ).scalar_one_or_none()

        if existing:
            # Update existing record
            self.session.execute(
                update(QTableModel)
                .where(QTableModel.state == state, QTableModel.action == action_hash)
                .values(qvalue=qvalue)
            )
        else:
            # Insert new record
            self.session.add(QTableModel(state=state, action=action_hash, qvalue=qvalue))
        self.session.commit()

    def select_best_action_db(self, state):
        try:
            best_action = self.session.execute(
                select(QTableModel.action).where(QTableModel.state == state)
                .order_by(QTableModel.qvalue.desc())
                .limit(1)
            ).scalar_one()
            return Position.from_hash(int(best_action))
        except NoResultFound:
            return None

    def select_max_q_value_db(self, state):
        max_qvalue = self.session.execute(
            select(func.max(QTableModel.qvalue)).where(QTableModel.state == state)
        ).scalar_one()
        return max_qvalue if max_qvalue is not None else self.default_q_value

    def select_states_count_db(self):
        count = self.session.execute(
            select(func.count(func.distinct(QTableModel.state)))
        ).scalar_one()
        return count

    def select_all_states_db(self):
        states = self.session.execute(
            select(QTableModel.state).distinct()
        ).scalars().all()
        return states

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
