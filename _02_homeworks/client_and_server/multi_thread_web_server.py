import socket
import threading

# 서버 IP와 포트 번호
HOST = '172.19.82.225'
PORT = 6789

def handle_request(request):
    headers = request.split('\n')
    file_requested = headers[0].split()[1]  # 파일 경로

    if file_requested == '/' or file_requested == '/HelloWorld.html':
        try:
            # HTML 파일 읽기
            with open('../../../../Desktop/자료/3-2/컴퓨터 네트워크/HW_1_2022136115_정희연/HelloWorld.html', 'r', encoding='utf-8') as file:
                content = file.read()

            response = 'HTTP/1.1 200 OK\n'
            response += 'Content-Type: text/html\n'
            response += '\n'
            response += content
        except FileNotFoundError:
            # 파일이 없을 때 404 응답
            response = 'HTTP/1.1 404 Not Found\n'
            response += 'Content-Type: text/html\n'
            response += '\n'
            response += '<html><body><h1>404 Not Found</h1></body></html>'
    else:
        # 다른 파일이면 404 응답
        response = 'HTTP/1.1 404 Not Found\n'
        response += 'Content-Type: text/html\n'
        response += '\n'
        response += '<html><body><h1>404 Not Found</h1></body></html>'

    return response

def client_handler(client_connection):
    request = client_connection.recv(1024).decode()
    print(f"Request: {request}")

    response = handle_request(request)

    client_connection.sendall(response.encode())
    client_connection.close()  # 클라이언트 연결 종료

# 소켓 설정하고 서버 시작
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)  # 최대 5개의 연결 대기 가능
    print(f"Server running on http://{HOST}:{PORT}")

    while True:
        # 클라이언트 연결 수락
        client_connection, client_address = server_socket.accept()
        print(f"Connection from {client_address}")

        # 새로운 스레드 생성
        thread = threading.Thread(target=client_handler, args=(client_connection,))
        thread.start()  # 스레드 시작
