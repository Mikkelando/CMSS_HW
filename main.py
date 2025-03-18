import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import imageio
import os

def generate_interarrival_times(rate, num_requests):
    return np.random.exponential(1/rate, num_requests)

def generate_service_times(mean, variance, num_requests):
    sigma = np.sqrt(variance)
    mu = mean
    return np.random.lognormal(mu, sigma, num_requests)

def simulate_queue(num_channels, queue_size, arrival_rate, service_mean, service_variance, num_requests=10000):
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





        # # Освобождаем серверы
        # servers = np.maximum(servers, current_time)  # Обновляем время освобождения серверов
        
        # # Проверяем простои
        # if np.all(servers == current_time):
        #     idle_time += current_time - last_free_time
        #     print(f"[{current_time:.2f}] Время простоя увеличено на {current_time - last_free_time:.2f}")
        
        # # Размещение заявки в сервер
        # while 0 in servers and queue:
        #     server_index = np.where(servers == 0)[0][0]
        #     service_time = queue.pop(0)
        #     servers[server_index] = current_time + service_time
        #     print(f"[{current_time:.2f}] Заявка из очереди отправлена на сервер {server_index+1}")
        
        # if 0 in servers:
        #     server_index = np.where(servers == 0)[0][0]
        #     servers[server_index] = current_time + service_times[i]
        #     last_free_time = servers[server_index]
        #     print(f"[{current_time:.2f}] Заявка {i+1} отправлена на сервер {server_index+1}")
        # elif len(queue) < queue_size:
        #     queue.append(service_times[i])
        #     print(f"[{current_time:.2f}] Заявка {i+1} добавлена в очередь. Очередь: {len(queue)}")
        # else:
        #     lost_requests += 1
        #     print(f"[{current_time:.2f}] Заявка {i+1} потеряна! Очередь заполнена.")
        
        # Логирование данных
        total_requests = sum(servers > current_time) + len(queue)
        log_data.append([current_time, sum(servers > current_time), len(queue), lost_requests, total_requests])
    
    log_df = pd.DataFrame(log_data, columns=["Time", "Busy_Servers", "Queue_Length", "Lost_Requests", "Total_Requests"])
    log_df.to_csv("queue_simulation_log.csv", index=False)
    
    print("Моделирование завершено.")
    print('ОЦЕНКА ВЕРОЯТНОСТИ ПРОСТОЯ: (counter): ', idle_counter / num_requests)
    print('ОЦЕНКА ВЕРОЯТНОСТИ ПРОСТОЯ: (time_durations): ', idle_time / arrival_times[-1])
    return idle_time / arrival_times[-1]

# Запуск моделирования
num_channels = 3
queue_size = 3
arrival_rate = 0.2
service_mean = 1
service_variance = 2
num_requests = 100000

prob_idle = simulate_queue(num_channels, queue_size, arrival_rate, service_mean, service_variance, num_requests)
print('ОЦЕНКА ВЕРОЯТНОСТИ ПРОСТОЯ: ', prob_idle)




# Визуализация через сохранение кадров
# print("Создание визуализации...")
# log_df = pd.read_csv("queue_simulation_log.csv")
# fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
# axes[0].set_ylabel("Занятые серверы")
# axes[1].set_ylabel("Очередь")
# axes[2].set_ylabel("Потерянные заявки")
# axes[3].set_ylabel("Всего заявок в системе")
# axes[3].set_xlabel("Время")

# if not os.path.exists("frames"): os.makedirs("frames")

# frame_files = []
# frame_skip = max(1, len(log_df) // 200)

# for frame in range(0, 300):
#     time = log_df["Time"][:frame]
    
#     axes[0].cla()
#     axes[1].cla()
#     axes[2].cla()
#     axes[3].cla()
    
#     axes[0].bar(time, log_df["Busy_Servers"][:frame], color='blue')
#     axes[1].bar(time, log_df["Queue_Length"][:frame], color='red')
#     axes[2].bar(time, log_df["Lost_Requests"][:frame], color='black')
#     axes[3].bar(time, log_df["Total_Requests"][:frame], color='green')
    
#     frame_path = f"frames/frame_{frame:04d}.png"
#     plt.savefig(frame_path)
#     frame_files.append(frame_path)
#     print(f"Сохранен кадр {frame}")

# print("Создание видео...")
# imageio.mimsave("queue_simulation.mp4", [imageio.imread(f) for f in frame_files], fps=30)
# print("Анимация сохранена в queue_simulation.mp4")
