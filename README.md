# Chat with database

## Start mySQL database with docker
```
$ docker run --name mysql-db -p 3306:3306 -e MYSQL_ROOT_PASSWORD=password@1234 -d mysql:9.3.0
```

Insert sample data
```
$ docker cp client_1.sql mysql-db:/client_1.sql
$ docker exec -it mysql-db bash

bash-5.1# mysql -u root -p

mysql> CREATE DATABASE IF NOT EXISTS client_1;
Query OK, 1 row affected (0.008 sec)

mysql> USE client_1;
Database changed

mysql> exit
Bye
bash-5.1# mysql -u root -p client_1 < /client_1.sql 
```