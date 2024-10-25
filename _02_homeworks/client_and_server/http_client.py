import socket
def http_client(host, port, filename):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # 서버와 TCP 연결
        sock.connect((host, port))
        # GET 요청 구성
        request = f"GET /{filename} HTTP/1.1\r\nHost: {host}\r\n\r\n"
        # 서버로 GET 요청 전송
        sock.sendall(request.encode())
        response = sock.recv(4096).decode()
        print(response)

if __name__ == "__main__":
    HOST = '172.19.82.225'
    PORT = 6789
    filename = "../../../../Desktop/자료/3-2/컴퓨터 네트워크/HW_1_2022136115_정희연/HelloWorld.html"

    http_client(HOST, PORT, filename)
