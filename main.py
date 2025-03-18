import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
# import imageio
import os

def generate_interarrival_times(rate, num_requests):
    return np.random.exponential(1/rate, num_requests)

def generate_service_times(mean, variance, num_requests):
    sigma = np.sqrt(variance)
    mu = mean
    return np.random.lognormal(mu, sigma, num_requests)

def simulate_queue_total(num_channels, queue_size, arrival_rate, service_mean, service_variance, num_requests=10000):
    print("Инициализация системы...")
    interarrival_times = generate_interarrival_times(arrival_rate, num_requests)
    service_times = generate_service_times(service_mean, service_variance, num_requests)
    
    arrival_times = np.cumsum(interarrival_times)
    servers = np.zeros(num_channels)  # Время окончания работы серверов
    queue = []
    idle_time = 0
    last_free_time = 0
    lost_requests = 0
    idle_counter = 0
    
    log_data = []
    
    for i in range(num_requests):
        current_time = arrival_times[i]
        print(f"[{current_time:.2f}] Обрабатываем заявку {i+1} из {num_requests}")

        out_flag = 0
        while not out_flag:
            t = min(current_time, *servers)

            if t != min(*servers):
                if len(queue) < queue_size:
                    queue.append(service_times[i])
                    print(f"[{current_time:.2f}] Заявка {i+1} добавлена в очередь. Очередь: {len(queue)}")
                    out_flag = 1
                else:
                    lost_requests += 1
                    print(f"[{current_time:.2f}] Заявка {i+1} потеряна! Очередь заполнена.")
                    out_flag = 1
            else:
                if len(queue) == 0:
                    if np.all(servers < current_time):
                        idle_counter += 1
                        idle_time += current_time - max(*servers)
                        print(f"[{current_time:.2f}] Время простоя увеличено на {(current_time - max(*servers)):.2f}")
                    j = min([k for k in range(len(servers)) if servers[k] < current_time])
                    servers[j] = current_time + service_times[i]
                    out_flag = 1
                    print(f"[{current_time:.2f}] Заявка {i+1} отправлена на сервер {j+1}")

                else:
                    j = np.argmin(servers)
                    service_time_tmp = queue.pop(0)
                    servers[j] += service_time_tmp
                    print(f"[{current_time:.2f}] Заявка из очереди отправлена на сервер {j+1}")





      
        
        # Логирование данных
        total_requests = sum(servers > current_time) + len(queue)
        log_data.append([current_time, sum(servers > current_time), len(queue), lost_requests, total_requests])
    
    log_df = pd.DataFrame(log_data, columns=["Time", "Busy_Servers", "Queue_Length", "Lost_Requests", "Total_Requests"])
    log_df.to_csv("queue_simulation_log.csv", index=False)
    
    print("Моделирование завершено.")
    print('ОЦЕНКА ВЕРОЯТНОСТИ ПРОСТОЯ: (counter): ', idle_counter / num_requests)
    print('ОЦЕНКА ВЕРОЯТНОСТИ ПРОСТОЯ: (time_durations): ', idle_time / arrival_times[-1])
    return idle_time / arrival_times[-1]

def simulate_queue_each(num_channels, queue_size, arrival_rate, service_mean, service_variance, num_requests=100, prev_ckpt = None):
    print("Инициализация системы...")
    interarrival_times = generate_interarrival_times(arrival_rate, num_requests)
    service_times = generate_service_times(service_mean, service_variance, num_requests)
    targets = [0 for i in range(num_requests)]
    
    arrival_times = np.cumsum(interarrival_times)
    if prev_ckpt:
        servers = prev_ckpt['servers']
        queue =  prev_ckpt['queue']
    else:
        servers = np.zeros(num_channels)  # Время окончания работы серверов
        queue = []
    idle_time = 0
    last_free_time = 0
    lost_requests = 0
    idle_counter = 0
    
    log_data = []
    
    for i in range(num_requests):
        current_time = arrival_times[i]
        # print(f"[{current_time:.2f}] Обрабатываем заявку {i+1} из {num_requests}")

        out_flag = 0
        while not out_flag:
            t = min(current_time, *servers)

            if t != min(*servers):
                if len(queue) < queue_size:
                    queue.append(service_times[i])
                    # print(f"[{current_time:.2f}] Заявка {i+1} добавлена в очередь. Очередь: {len(queue)}")
                    out_flag = 1
                else:
                    lost_requests += 1
                    # print(f"[{current_time:.2f}] Заявка {i+1} потеряна! Очередь заполнена.")
                    out_flag = 1
            else:
                if len(queue) == 0:
                    if np.all(servers < current_time):
                        idle_counter += 1
                        idle_time += current_time - max(*servers)
                        # print(f"[{current_time:.2f}] Время простоя увеличено на {(current_time - max(*servers)):.2f}")
                        targets[i] = 1
                    j = min([k for k in range(len(servers)) if servers[k] < current_time])
                    servers[j] = current_time + service_times[i]
                    out_flag = 1
                    # print(f"[{current_time:.2f}] Заявка {i+1} отправлена на сервер {j+1}")

                else:
                    j = np.argmin(servers)
                    service_time_tmp = queue.pop(0)
                    servers[j] += service_time_tmp
                    # print(f"[{current_time:.2f}] Заявка из очереди отправлена на сервер {j+1}")





      
        
        # Логирование данных
        total_requests = sum(servers > current_time) + len(queue)
        log_data.append([current_time, sum(servers > current_time), len(queue), lost_requests, total_requests])
    
    log_df = pd.DataFrame(log_data, columns=["Time", "Busy_Servers", "Queue_Length", "Lost_Requests", "Total_Requests"])
    log_df.to_csv("queue_simulation_log.csv", index=False)
    
    print("Моделирование завершено.")
    last_ckpt = {'servers' : servers, 'queue':queue}
    return idle_time / arrival_times[-1], targets, last_ckpt





import math

def normal_clt_draw(n=12):
    """
    Генерирует одно (приблизительно) N(0,1) число,
    используя ЦПТ и равномерные [0,1] случайные gamma_i.
    """
    s = 0.0
    for _ in range(n):
        gamma_i = np.random.rand()  # [0,1)
        s += (gamma_i - 0.5)
    return math.sqrt(12.0 / n) * s


def cdf_mc(z, sample_size=10_000, n_clt=12):
    """
    Оценивает CDF(z) для X ~ N(0,1), 
    генерируя sample_size реализаций через normal_clt_draw(n_clt).
    Возвращает долю, которая <= z.
    """
    count_le = 0
    for _ in range(sample_size):
        x = normal_clt_draw(n_clt)
        if x <= z:
            count_le += 1
    return count_le / sample_size


def normal_ppf_clt(alpha, sample_size=10_000, n_clt=12, tol=1e-3):
    """
    Ищем z_alpha, т.е. z, при котором CDF(z)=alpha 
    для стандартного нормального, 
    используя ЦПТ+Монте–Карло+метод половинного деления.
    
    :param alpha: (0,1) уровень квантиля
    :param sample_size: сколько раз генерировать для оценки CDF(z)
    :param n_clt: параметр в normal_clt_draw
    :param tol: допустимая точность по CDF 
               (или по смещению z, можно усложнить)
    """
    if alpha <= 0.0:
        return -1e9
    if alpha >= 1.0:
        return 1e9
    
    left, right = -10.0, 10.0
    while (right - left) > 1e-3:  # шаг в пространстве z
        mid = 0.5*(left + right)
        f_mid = cdf_mc(mid, sample_size, n_clt)  # оценка Phi(mid)
        if f_mid < alpha:
            left = mid
        else:
            right = mid
    return 0.5*(left + right)


def main():
    eps = 0.05
    alpha = 0.9
    z_est = normal_ppf_clt(alpha, sample_size=50000, n_clt=30, tol=1e-4)
    print(f"Квантиль уровня alpha={alpha:.3f} ≈ {z_est:.4f}")

    

    # Запуск моделирования
    num_channels = 3
    queue_size = 3
    arrival_rate = 0.2
    service_mean = 1
    service_variance = 2
    num_requests = 10
    N = num_requests

    # prob_idle = simulate_queue_total(num_channels, queue_size, arrival_rate, service_mean, service_variance, num_requests)

    prob_idle, targets, ckpt = simulate_queue_each(num_channels, queue_size, arrival_rate, service_mean, service_variance, num_requests)
    targets = np.array(targets)
    E = np.mean(targets)
    E2 = np.mean(targets**2)
    D = E2 - E**2
    N_new = int ( (z_est)**2 * D/ (eps ** 2) )
    
    print('ITERATIONS: ', N)
    while N < N_new:
        num_requests = N_new - N
        prob_idle, targets, ckpt = simulate_queue_each(num_channels, queue_size, arrival_rate, service_mean, service_variance, num_requests)
        targets = np.array(targets)
        E = np.mean(targets)
        E2 = np.mean(targets**2)
        D = E2 - E**2
        N = N_new
        N_new = int ( (z_est)**2 * D/ (eps ** 2) )

    print('N FOUND: ', N)
    print('ИСКОМАЯ ВЕЛИЧИНА: ', E)



if __name__ == "__main__":
    
    main()

