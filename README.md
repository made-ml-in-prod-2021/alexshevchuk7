Homework4

Самооценка:
1. Сервис Кубернетес поднят на Google Cloud (картинка ниже):
![Kub Google cloud](https://github.com/made-ml-in-prod-2021/alexshevchuk7/blob/homework4/kubernetes/cluster-success.png)
Результат комманды kubectl cluster-info:
![kubectl cluster-info](https://github.com/made-ml-in-prod-2021/alexshevchuk7/blob/homework4/kubernetes/kubectl%20cluster-info.png)
2. Поднято приложение с простым манифестом:
![kubectl apply](https://github.com/made-ml-in-prod-2021/alexshevchuk7/blob/homework4/kubernetes/kubectl-apply-resources.png)
3. Подготовлен манифест с реквестами и лимитами.
Реквесты позволяют Kubernetes определить минимальный объем CPU и RAM. Контейнер с прописанными реквестами размещается
в приоритетном порядке. Лимиты позволяют управлять максимальным объемом CPU и RAM, не позволяя контейнеру потребить большее 
количество ресурсов, которое может быть необходимым для работы другим конейнерам. 
4. Readiness probe считывает метку готовности приложения к работе. Liveness проверяет живо ли приложение и если нет, то перезапускает
контейнер. 
Измененное приложение находится в файле app_failure.py.
Я установил задержку в 30 секунд при загрузке модели и через минуту после загрузки вызывал несуществующую функцию, чтобы взывать
ее падение. В результате контейнер перезапускался. Правда это произошло только 5 раз и я не уверен, все ли сделал правильно. 
Liveness probe также обращался к пути /health, но если конейнер уже лежит, то этого пути не существует. Возможно, что в этом ошибка.
5. Итого: 5 + 4 + 2 + 3(здесь не уверен, что правильно прописал livenessProbe - на усмотрение reviewers)
