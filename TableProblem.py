import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from pandas import to_datetime


def read_file(file):
    df = pd.read_excel(file)
    df_dt = to_datetime(df.time, format="%Y/%m/%d")
    s_d = df_dt.dt.day
    s_h = df_dt.dt.hour
    s_m = df_dt.dt.minute
    s_s = df_dt.dt.second
    df['date'] = s_d
    df['arrival'] = s_h * 60 + s_m + s_s / 60
    df.drop(['time'], axis=1, inplace=True)
    df['served'] = 1
    category = df.ps.value_counts()
    return df, category


def match(available):
    if sum(available) == 0:
        index = -1
    else:
        p = [i / sum(available) for i in available]
        p = np.array(p)
        index = np.random.choice(range(len(available)), p=p.ravel())
    return index


def update_state(table_state, table_choose, server_state):
    for i in range(len(table_choose)):
        table_i_state = table_state.iloc[table_choose[i]]
        exchange = [int(table_i_state[4])]
        seat = [i for i in table_i_state[0:4]]
        index = []
        for j, v in enumerate(seat):
            if v > 0:
                seat[j] = 1
            else:
                index.append(j)
        if sum(seat) == 2:
            flag = sum(index) % 2
            state = 5 if flag == 0 else 2
        else:
            state = 4 - sum(seat)
            state = int(state)
        exchange.append(state)
        server_state[exchange[0]] -= 1
        server_state[exchange[1]] += 1
        table_state.iloc[table_choose[i], 4] = state
    return table_state, server_state


def update_state_departure(last_date, current_date, row, table_state):
    table_choose = []
    if current_date != last_date:
        table_state.loc[:, 0:4] = 0
        table_choose = [i for i in range(10)]
    else:
        arrival = row.arrival
        for i in range(10):
            for j in range(4):
                item = table_state.iloc[i, j]
                if item != 0 and item <= arrival:
                    table_state.iloc[i, j] = 0
                    table_choose.append(i)
    return table_state, table_choose


class Customer:
    def __init__(self, customer_type, start=0, money=0):
        self.money = money
        self.customer_type = customer_type
        self.start = start

    def choose_server(self, server_state):
        choose = -1
        if server_state[4] > 0:  # 如果有空桌，不管顾客是多少人，一定入座
            choose = 4
        else:
            pick = np.random.uniform()  # 没有空桌，以30%的概率直接离开
            if pick <= 0.7:
                if self.customer_type == 1:
                    available_server = [server_state[5], server_state[1], server_state[2], server_state[3]]
                    # 没有空桌的情况下，如果是1个人则可以在n5和n1,n2,n3中选
                    index = match(available_server)
                    if index == 0:
                        choose = 5
                    else:
                        choose = index
                elif self.customer_type == 2:
                    available_server = [server_state[2], server_state[3]]
                    index = match(available_server)
                    choose = index + 2 if index >= 0 else -1
                elif self.customer_type == 3:
                    choose = 3 if server_state[3] > 0 else -1
        return choose, server_state

    def choose_seat(self, choose, table_state):
        table = 0
        while True:
            table_i_state = [i for i in table_state.iloc[table]]
            if table_i_state[4] == choose:
                break
            table += 1
        eating_time = np.random.uniform(20, 40)  # 产生随机数表示用餐时间
        finish_time = self.start + eating_time  # 记录离开座位的时间
        table_choose = table  # 记录该顾客选择的餐桌
        seat_state = [j for j in table_i_state[0:4]]  # 只取前四列表示座位状态
        state_before = table_i_state[4]
        ps = self.customer_type
        ps = int(ps)
        available = []
        for index, value in enumerate(seat_state):
            if value == 0:
                available.append(index)
        seat_choose = []
        if ps != 2 or ps == len(available):
            seat_choose = random.sample(available, ps)
        else:
            for j in range(len(available)):
                if available[j+1]-available[j] == 1:
                    seat_choose = [available[j], available[j+1]]
                    break
        for i in range(ps):
            index = int(seat_choose[i])
            seat_state[index] = finish_time
        seat_state.append(state_before)
        table_state.iloc[table_choose] = seat_state
        table_choose = [table_choose]
        return table_state, table_choose

    @staticmethod
    def abandon(customer_type, money, loss, loss_type):
        customer_type = int(customer_type)
        loss_type[customer_type - 1] += 1
        loss += money
        loss = round(loss, 2)
        return loss, loss_type


def main():
    loss = 0
    loss_type = [0]*4
    loss_rate = []
    table_state = pd.DataFrame(np.zeros([10, 4]))  # 构建初始的餐桌状态
    table_state['状态'] = [4] * 10
    server_state = [0, 0, 0, 0, 10, 0]  # server_state形如[n0,n1,n2,n3,n4,n5] list 1*6 n5表示对角线两个人
    file = 'Data-Problem3.xlsx'
    df, category = read_file(file)
    last_date = 1
    day = 0
    order_loss = 0
    for i in range(df.shape[0]):
        row = df.iloc[i]
        current_date = row.date
        table_state, table_choose = update_state_departure(last_date, current_date, row, table_state)
        if len(table_choose):
            table_state, server_state = update_state(table_state, table_choose, server_state)
        new_customer = Customer(row.ps, row.arrival, row.money)
        choose, server_state = new_customer.choose_server(server_state)
        if choose == -1:
            loss, loss_type = new_customer.abandon(new_customer.customer_type, new_customer.money, loss, loss_type)
            order_loss += 1
            df.iloc[i, -1] = 0
        else:
            table_state, table_choose = new_customer.choose_seat(choose, table_state)
            table_state, server_state = update_state(table_state, table_choose, server_state)
        if current_date != last_date:
            day += 1
            print("第%d天" % day)
            print("Total Loss:")
            print(loss)
            print(loss_type)
        last_date = current_date
        loss_rate.append(order_loss/(i+1))
    print("第100天")
    print("Total Loss")
    print(loss)
    print(loss_type)
    print("\n")
    for i in range(4):
        loss_type[i] = round(loss_type[i]/category[i+1], 4)
    print(loss_type)

    plt.plot(range(df.shape[0]), loss_rate)
    plt.xlabel('number of order')
    plt.ylabel('loss rate')
    plt.show()

    ps_ser = pd.crosstab(df.ps, df.served)
    ps_ser[[0, 1]].plot(kind='bar', stacked=True)
    plt.show()


if __name__ == '__main__':
    main()
