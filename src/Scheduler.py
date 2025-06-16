import pickle
import time
from tqdm import tqdm
import torch
import cv2
import pika
from src.Model import SplitDetectionPredictor # trong RpcClient.py mới sử dụng chia model thật còn ở đây truyền inference qua từng phần

class Scheduler: # mục đích là tính thời gian inference trên từng phần head, mid, tail, sắp xếp giao tiếp các client với nhau và với server
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id # id của client
        self.layer_id = layer_id # id chỉ vị trí của client là nằm đầu hay sau hay giữa của model
        self.channel = channel 
        self.device = device
        self.intermediate_queue = f"intermediate_queue_{self.layer_id}" # hàng đợi này là của cái layer_id sẽ gửi dữ liệu đến đây
        self.channel.queue_declare(self.intermediate_queue, durable=False) # chính là khai báo 1 hàng đợi dành riêng cho từng layer_id
        # durable = false có nghĩa là sẽ không lưu lại tin nhắn trong hàng đợi, mà tin nhắn gửi đến bên kia sẽ dùng ngay

    def send_next_part(self, intermediate_queue, data, logger): # mục đích là gửi dữ liệu đến layer tiếp theo
        if data != 'STOP': # tức là chưa dừng lại thì
            data["modules_output"] = [t.cpu() if isinstance(t, torch.Tensor) else None for t in data["modules_output"]]
            # data tại vị trí layers_output chứa ds các đầu ra của module quan trọng, còn không quan trọng thì là None
            # nó duyệt từng module quan trọng trong layers_output, nếu là tensor thì chuyển về cpu, còn không thì cho là None
            # sau đó lưu lại vào data ở vị trí layers_output
            message = pickle.dumps({ # chính là đóng gói lại dữ liệu để gửi đi layer khác rồi lưu thành message, dữ liệu
            # sẽ gồm các đầu ra của module quan trọng
                "action": "OUTPUT",
                "data": data
            })

            self.channel.basic_publish( # mục đích chính là xuất vào hàng đợi ở trên, layer khác sẽ vào hang đợi này để lấy dữ liệu
                exchange='',
                routing_key=intermediate_queue,# đây là tên hàng đợi ứng với từng layer_id
                body=message,
            )
        else: # còn nếu nhận được tin nhắn dừng lại thì
            message = pickle.dumps(data) # vẫn đóng gói dữ liệu lần cuối
            self.channel.basic_publish( # sau đó lại xuất vào hàng đợi đó tiếp, tương ứng với layer_id
                exchange='',
                routing_key=intermediate_queue,
                body=message,
            )

    def inference_head(self, model, data, batch_frame, logger): # mục đích là lấy video đầu vào, rồi cho chạy qua head

        time_inference = 0 # thời gian inference
        i=1
        count=0
        similarity_tensor=[]
        predictor = SplitDetectionPredictor(model, overrides={"imgsz": 640}) # đây là lớp đưa đầu ra của model vào để xử lý

        model.eval()
        model.to(self.device)
        video_path = data
        cap = cv2.VideoCapture(video_path) # cap chính là cái video sau khi mở ra từng path
        if not cap.isOpened():
            logger.log_error(f"Not open video")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS) # frame per second, chính là số khung hình của video lấy trong 1s
        logger.log_info(f"FPS input: {fps}")
        path = None
        pbar = tqdm(desc="Processing video (while loop)", unit="frame") # sẽ hiển thị ra cho mình xem % hoàn thành video

        # if self.layer_id==1:
        #     timer = time.time()
        #     timer = {"timer": timer, "layer_id":1}
        #     self.connect()
        #     self.channel.queue_declare('timer_queue', durable=False) # chính là cái hàng đợi gửi lên để bên server lắng nghe
        #     self.channel.basic_publish(exchange='', # chỉ là gửi tin nhắn đến hàng đợi trên thôi
        #                         routing_key='timer_queue',
        #                         body=pickle.dumps(timer))
        while True: # vòng lặp này là để lấy từng khung hình trong video lặp cho đến hết video, hiểu từng khung hình là từng cái hình thôi
            # còn video là lắp ghép của rất nhiều hình lại
            start = time.time()
                
            ret, frame = cap.read() # ret là return nếu trả về true tức khung frame được đọc thành công
            # còn frame sẽ là mảng 3 chiều lưu tính chất của từng pixel gồm height, width và màu

            if not ret and reference_tensor is not None: # nếu không đọc được
                list_reference_tensor=[]
                list_reference_tensor.append(reference_tensor)
                reference_tensor = torch.stack(list_reference_tensor)
                reference_tensor = reference_tensor.to(self.device)
                predictor.setup_source(reference_tensor)
                for predictor.batch in predictor.dataset:
                    path, reference_tensor, _ = predictor.batch
                preprocess_image = predictor.preprocess(reference_tensor)
                y = model.forward_head(preprocess_image)
                time_inference += (time.time() - start)
                self.send_next_part(self.intermediate_queue, y, logger)
                reference_tensor = None
                pbar.update(batch_frame)

            if not ret and reference_tensor is None: # nếu không đọc được
                y = 'STOP'
                self.send_next_part(self.intermediate_queue, y, logger) # thì gửi tin nhắn stop lên hàng đợi để layer kia biết là 
                # không có video, sau đó kết thúc
                break
            frame = cv2.resize(frame, (640, 640)) # đây chính là mảng 3 chiều lưu các pixel của từng khung hình trong video
            # bây giờ thay đổi đúng kích thước đầu vào
            tensor = torch.from_numpy(frame).float().permute(2, 0, 1)  # chuyển từ mảng thành tensor nhưng lại có dạng (0-255), 
            # trong khi pytoch cần float (0.0-255.0) nên cần chuyển thành float, chuyển thứ tự height, width, kênh màu thành
            #  kênh màu, height, width, nên sẽ có shape là (3, 640, 640)
            tensor /= 255.0 # mỗi pixel có giá trị 0.0-255.0 nên chia 255.0 để đưa về khoảng 0.0-1.0
            if i==1:
                reference_tensor=tensor
                similarity_tensor.append(reference_tensor)
                i+=1
                continue
            similarity = torch.nn.functional.cosine_similarity(tensor.flatten(), reference_tensor.flatten(), dim=0) # so sánh
            count+=1
            logger.log_info(f"similarity: {similarity} - {count}")
            
            if similarity >0.94:
                similarity_tensor.append(tensor) # nếu gần giống thì thêm vào

            else: # nếu không tức đã đã, lúc này lấy tensor trung bình rồi truyền đi
                #if len(similarity_tensor) ==0:

                if len(similarity_tensor)!=0:
                    average_tensor = torch.mean(torch.stack(similarity_tensor), dim=0) # tính trung bình các tensor trong similarity_tensor
                    list_average_tensor=[]
                    list_average_tensor.append(average_tensor)
                    average_tensor = torch.stack(list_average_tensor)
                    average_tensor = average_tensor.to(self.device) # chuyển tensor đến device để tính toán, cụ thể ở đây là cuda
                    # cuda chính là nền tảng tính toán song song trên GPU do NVIDIA phát triển
                    similarity_tensor =[]
                    similarity_tensor.append(tensor)
                    reference_tensor=tensor
                    # chuẩn bị dữ liệu
                    # logger.log_info(f"---------average: {average_tensor.shape} ---------")
                    predictor.setup_source(average_tensor) # chuẩn bị dữ liệu cho model, cụ thể là đầu vào của model
                    # hàm này sẽ đọc dữ liệu đầu vào, sau đó tiền xử lý resize về 640x640 như yêu cầu của tham số đầu vào của predictor
                    for predictor.batch in predictor.dataset: # mỗi batch hay mỗi lô hàng trong dataset chính là 1 khung hình, vì input_image
                    # là lưu nhiều khung hình, vì nó sẽ lặp qua các khung hình rồi lưu vào input_image, và 1 lần sẽ xử lý cùng lúc input_image
                    # với số lượng là batch_frame
                        path, average_tensor, _ = predictor.batch # mỗi bacth hay lô hàng chính là 1 khung hình sẽ có path, các pixel ở dạng tensor
                    # và các pixel ở dạng mảng numpy
                    # kết quả sẽ được: input_image sẽ lưu các pixel dạng tensor của từng khung hình cần xử lý cùng lúc, nó sẽ là list
                
                    # tiền xử lý ảnh sau khi input_image đã lưu các pixel của từng khung hình 
                    preprocess_image = predictor.preprocess(average_tensor) # đưa các pixel đó vào tiền xử lý sẽ được các tensor của từng khung
                    # quá trình này sẽ gồm resize, chuẩn hóa pixel, chuyển thành kênh màu, height, width

                    # ảnh sau khi xử lý xong sẽ đưa qua model ở phần head trước
                    y = model.forward_head(preprocess_image) # save_layers chính là các module quan trọng cần lưu lại đầu ra
                    # kết quả sẽ được y là dạng key-value, layers_output sẽ là 1 list chứa các đầu ra của các module quan trọng, còn lại None
                    # còn last_layer_idx chính là vị trí cuối cùng trong ds y, tức là module cuối cùng trong phần head

                    time_inference += (time.time() - start) # tính thời gian inference cho mỗi batch_frame, chính là cùng lúc số khung hình
                    # cái này là cộng lại để lấy tổng thời gian là bao nhiêu
                    
                    # if self.layer_id==1:
                    #     timer = time.time()
                    #     timer = {"timer": timer, "layer_id":1}
                    #     self.connect()
                    #     self.channel.queue_declare('timer_queue', durable=False) # chính là cái hàng đợi gửi lên để bên server lắng nghe
                    #     self.channel.basic_publish(exchange='', # chỉ là gửi tin nhắn đến hàng đợi trên thôi
                    #             routing_key='timer_queue',
                    #             body=pickle.dumps(timer))
                
                    self.send_next_part(self.intermediate_queue, y, logger) # đưa dữ liệu đầu ra từ phần head lên hàng đợi
                    # để phần mid lên đó lấy dữ liệu để chạy inference tiếp

                    pbar.update(batch_frame) # cập nhật lại thanh tiến độ, tức là đã xử lý xong batch_frame khung hình rồi, nó sẽ cộng vào
                    # sẽ hiển thị % hoàn thành theo thời gian thực

        cap.release()
        pbar.close()
        logger.log_info(f"End Inference Head.")
        return time_inference

    def inference_mid(self, model, batch_frame, logger):
        time_inference = 0
        
        model.eval()
        model.to(self.device)
        last_queue = f"intermediate_queue_{self.layer_id - 1}" # - 1 có nghĩa là nếu đang 2 tức nó là phần giữa model muốn lấy phần đầu thì phải -1
        self.channel.queue_declare(queue=last_queue, durable=False) # khai báo tạo ra hàng đợi
        self.channel.basic_qos(prefetch_count=50)

        pbar = tqdm(desc="Processing video (while loop)", unit="frame")

        while True:
            method_frame,_, body = self.channel.basic_get(queue=last_queue, auto_ack=True) #truy cập vào hàng đợi phía trên
            # để lấy tin nhắn, chính là đầu ra của từ các module ở phần head, liên tục lấy tin nhắn về vì có vòng while
            if method_frame and body:

                received_data = pickle.loads(body)
                if received_data != 'STOP':
                    y = received_data["data"] # chính là đầu ra dạng key-value
                    y["modules_output"] = [t.to(self.device) if t is not None else None for t in y["modules_output"]] # key này chính là đầu ra
                    start = time.time()
                    # chạy trên phần mid
                    y = model.forward_mid(y) # kết quả chính là dạng key-value

                    time_inference += (time.time() - start)
                    
                    self.send_next_part(self.intermediate_queue, y, logger)

                    pbar.update(batch_frame) # cập nhật rồi hiển thị thanh %
                else:
                    break
            else:
                continue

        y = 'STOP'
        self.send_next_part(self.intermediate_queue, y, logger)

        pbar.close()
        logger.log_info(f"End Inference Mid.")
        return time_inference

    def inference_tail(self, model, batch_frame, logger): # không có data và save_layers ở đây, vì đến cuối rồi, nó chỉ lấy dữ liệu từ hàng đợi thôi
        time_inference = 0

        model.eval()
        model.to(self.device)
        last_queue = f"intermediate_queue_{self.layer_id - 1}" # 
        self.channel.queue_declare(queue=last_queue, durable=False) # khai báo tạo ra hàng đợi
        self.channel.basic_qos(prefetch_count=50) #qos là quality of service, số lượng message chưa ack tối đa mà consumer có thể xử lý,
        # ở đây message chưa ack - acknowledge (hiểu đơn giản là chưa được xác nhận đã xử lý xong, nghĩa là message vẫn đang xử lý
        # trên consumer đó)
        pbar = tqdm(desc="Processing video (while loop)", unit="frame") # tiếp tục sẽ là thanh để hiển thị % hoàn thành video

        while True:
            method_frame, header_frame, body = self.channel.basic_get(queue=last_queue, auto_ack=True) #truy cập vào hàng đợi phía trên
            # để lấy tin nhắn, chính là đầu ra của từ các module ở phần head
            if method_frame and body:

                received_data = pickle.loads(body)
                if received_data != 'STOP':
                    y = received_data["data"] # chính là đầu ra dạng key-value
                    y["modules_output"] = [t.to(self.device) if t is not None else None for t in y["modules_output"]] # key này chính là đầu ra
                    start = time.time()
                    # if self.layer_id==3:
                    #     timer = time.time()
                    #     timer = {"timer": timer, "layer_id":3}
                    #     self.connect()
                    #     self.channel.queue_declare('timer_queue', durable=False) # chính là cái hàng đợi gửi lên để bên server lắng nghe
                    #     self.channel.basic_publish(exchange='', # chỉ là gửi tin nhắn đến hàng đợi trên thôi
                    #             routing_key='timer_queue',
                    #             body=pickle.dumps(timer))
                    # chạy trên phần tail
                    predictions = model.forward_tail(y) # kết quả chính là dạng key-value
                    # cái dự đoán này là cho từng frame tức là dự đoán cho từng khung hình trong video và nó có 3 scale khác nhau, chứ không phải cả video

                    # chạy hậu xử lý từ kết quả đầu ra của model, tức là xem kết quả đầu ra
                    # if save_output:
                    #     results = predictor.postprocess(predictions, y["img"], y["orig_imgs"], y["path"])
                    # ở trên là sẽ bổ sung thêm mỗi tin nhắn gửi lên hàng đợi ngoài đầu ra của các layer thì cần img, orig_imgs, path để xem kqua

                    time_inference += (time.time() - start)
                    pbar.update(batch_frame) # cập nhật rồi hiển thị thanh %
                else:
                    break
            else:
                continue
        pbar.close()
        logger.log_info(f"End Inference Tail.")
        return time_inference

    def inference_func(self, model, data, num_layers, batch_frame, logger):
        time_inference = 0
        if self.layer_id == 1:# tức là client nó nằm đầu model
            time_inference = self.inference_head(model, data, batch_frame, logger)
        elif self.layer_id == num_layers:# tức là client nằm ở cuối model
            time_inference = self.inference_tail(model, batch_frame, logger)
        else:
            time_inference = self.inference_mid(model, batch_frame, logger) # nếu không thì nó nằm giữa model
        return time_inference

    def connect(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.address, 5672, self.virtual_host, credentials))
        self.channel = self.connection.channel()