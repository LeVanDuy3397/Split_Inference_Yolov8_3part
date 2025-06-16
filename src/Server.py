import os
import sys
import base64
import pika
import pickle
import torch
import torch.nn as nn

import src.Model
import src.Log


class Server:
    def __init__(self, config):
        # RabbitMQ
        address = config["rabbit"]["address"] # truy cập vào RabbitMQ và bên trong là address
        username = config["rabbit"]["username"]
        password = config["rabbit"]["password"]
        virtual_host = config["rabbit"]["virtual-host"]

        self.model_name = config["server"]["model"] # truy cập vào server và bên trong là model
        self.total_clients = config["server"]["clients"]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        self.cut_module = config["server"]["cut-module"]
        self.batch_frame = config["server"]["batch-frame"]

        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')
        self.channel.queue_declare(queue='timer_queue')

        self.register_clients = [0 for _ in range(len(self.total_clients))] # chạy qua 3 client, rồi tạo ra 1 list [0,0]
        self.list_clients = []

        self.channel.basic_qos(prefetch_count=1)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)
        self.channel.basic_consume(queue='timer_queue', on_message_callback=self.log)
        # bắt đầu lắng nghe tin nhắn từ hàng đợi rpc_queue, nếu có tin nhắn thì gọi đến hàm on_request rồi truyền vào các tham số
        # hình dung hàng đợi này là 1 cái kênh và các client sẽ gửi tin nhắn đến đây, rồi on_request sẽ xử lý tin nhắn đó

        self.data = config["data"] # truy cập vào data trong file cấu hình
        self.debug_mode = config["debug-mode"] # truy cập vào debug-mode trong file cấu hình
        
        log_path = config["log-path"] # truy cập vào log-path trong file cấu hình
        self.logger = src.Log.Logger(f"{log_path}/app.log") # tạo ra 1 logger mới từ class logger với đường dẫn là log_path + app.log
        # nằm trong class logger
        self.logger.log_info(f"Application start. Server is waiting for {self.total_clients} clients.")
        # đây là 1 phương thức trong class logger, in ra thông báo là server đang chờ đợi các client kết nối đến

    def on_request(self, ch, method, props, body): # mục đích là xử lý yêu cầu từ client gửi đến server, server sẽ đăng ký client 
        # và kiểm tra số lượng đã đủ chưa 
        # tham số của 1 yêu cầu: ch là kênh, methosd là phương thức gửi, properties là thuộc tính, body là nội dung tin nhắn
        message = pickle.loads(body) # tất cả cái này là thông tin từ client đưa đến nằm trong hàng đợi rpc_queue
        action = message["action"]
        client_id = message["client_id"] # đây là id của cái client chạy 1 phần của model
        layer_id = message["layer_id"] # id của layer khi chia model thành 3 layer

        if action == "REGISTER":
            if (str(client_id), layer_id) not in self.list_clients: # id của client với id của layer lớn chia từ model mà không có
                self.list_clients.append((str(client_id), layer_id)) # thì thêm client đó vào danh sách clients gồm đki với k đki
            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue") # in ra tin nhắn từ client
            self.register_clients[layer_id-1] += 1 # tức client chạy phần đầu là 1 thì lưu vào list thì lấy chỉ số 0

            if self.register_clients == self.total_clients: # kiểm tra client đăng ký đủ số lượng client ban đầu
                src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                self.notify_clients() # nếu như đủ số lượng client thì gửi thông tin đến tất cả client để bắt đầu
        else:
            timer = message["timer"]
            src.Log.print_with_color(f"Time: {timer}", "green")

        ch.basic_ack(delivery_tag=method.delivery_tag) # trong method có delivery_tag, đây là 1 cái id để xác nhận cái tin nhắn 
        # trong hàng đợi, đưa tin nhắn đó vào hàm basic_ack để xác nhận đã nhận được tin nhắn đó rồi là ok

    def log(self, ch, method, props, body): # mục đích là xử lý yêu cầu từ client gửi đến server, server sẽ đăng ký client 
        # và kiểm tra số lượng đã đủ chưa 
        # tham số của 1 yêu cầu: ch là kênh, methosd là phương thức gửi, properties là thuộc tính, body là nội dung tin nhắn
        message = pickle.loads(body) # tất cả cái này là thông tin từ client đưa đến nằm trong hàng đợi rpc_queue
        timer = message["timer"]
        layer_id = message["layer_id"]
        src.Log.print_with_color(f"Time: {timer} of layer_id: {layer_id}", "green")
        ch.basic_ack(delivery_tag=method.delivery_tag) 

    def send_to_response(self, client_id, message):
        reply_queue_name = f"reply_{client_id}" # trả lời lên hàng đợi của riêng client đó
        
        self.reply_channel.queue_declare(reply_queue_name, durable=False) # khai báo 1 hàng đợi mới để
        # trả lời lại với tên hàng đợi là tên của client, không bền vững vì durable=False 
        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red") # thông báo

        self.reply_channel.basic_publish( # xuất vào hàng đợi đó, gồm key và tin nhắn
            exchange='',
            routing_key=reply_queue_name, # phải gửi đến đúng tên hàng đợi của client đã khai báo ở trên
            body=message
        )

    def start(self):
        self.channel.start_consuming() # bắt đầu tiêu thụ hay lắng nghe các tin nhắn trong hàng đợi rpc_queue đối với server

    def notify_clients(self): # gửi các thông tin quan trọng đến cho các client

        default_splits = { # đây chỉ là 1 cái mặc định thôi, có thể thay đổi tùy theo ý muốn
            "a": (1, 9), 
            "b": (9, 16), 
            "c": (16, 22) 
        }

        splits = default_splits[self.cut_module] # ở đây cut_layer đang là a, tức lấy hàng đầu, lấy thông tin các điểm cắt

        file_path = f"{self.model_name}.pt"
        if os.path.exists(file_path): # nếu file đó tồn tại
            src.Log.print_with_color(f"Load model {self.model_name}.", "green") # cái này chỉ là ỉn ra thôi
            with open(f"{self.model_name}.pt", "rb") as f: # rb là read binary, with chỉ là file tự đóng khi kết thúc
            # hiểu đơn giản f chính là file đó
                file_bytes = f.read() # đọc file, file này chính là chuỗi byte
                encoded = base64.b64encode(file_bytes).decode('utf-8') 
            # từ file .pt mở ra dạng byte sau đó mã hóa chuyển thành dạng base64 sau đó giải mã chuyển thành utf-8 dễ đọc hơn
        else: # nếu file không tồn tại, thì thông báo rồi thoát
            src.Log.print_with_color(f"{self.model_name} does not exist.", "yellow")
            sys.exit()

        for (client_id, _) in self.list_clients:

            response = {"action": "START",
                        "message": "Server accept the connection",
                        "model": encoded, # lấy file model yolov8n đã mã hóa ở trên rồi truyền vào các client
                        "split_module1": splits[0],
                        "split_module2": splits[1], 
                        "batch_frame": self.batch_frame, # chính là 1 là mỗi lần xử lý cùng lúc bao nhiêu khung hình
                        "num_layers": len(self.total_clients), # số lượng layer trong model chính bằng với số lượng client, vì chia thành 2 layer
                        # hiểu là gom toàn bộ layer con thành 1 layer lớn, thì đây là bao nhiêu layer lớn
                        "model_name": self.model_name, # chính là tên model 
                        "data": self.data, # chính là file video cần nhận diện và mình sẽ đưa nó đến cho client đầu nhận rồi chạy inference
                        "debug_mode": self.debug_mode} # chế độ gỡ lỗi

            self.send_to_response(client_id, pickle.dumps(response))
