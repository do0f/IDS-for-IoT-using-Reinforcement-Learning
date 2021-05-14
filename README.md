# IDS-for-IoT-using-Reinforcement-Learning

- Проект собирается с помощью средств CMake
- Для проекта необходима установка библиотеки LibTorch, инструкция по установке: https://pytorch.org/cppdocs/installing.html \
и CUDA Toolkit, https://developer.nvidia.com/cuda-downloads
- Для корректной работы необходимо передать в командную строку режим работы программы ("training", "testing") и корректный путь к нормализованному датасету в формате .csv.
При режиме работы "training" после окончания обучения в директории создастся файл "policyNet.pt", содержащий веса для обученной нейронной сети.
При режиме работы "testing" после окончания тестирования в директории создастся файл "testResults.txt", содержащий метрики результатов тестирования.
