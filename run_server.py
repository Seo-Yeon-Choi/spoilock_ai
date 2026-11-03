"""Windows 에서 FastAPI 서버 실행 (WinError 64 'Accept failed on a socket' 방지)

  python run_server.py
  종료: Ctrl+C (1~2초 안에 'Application shutdown complete' 가 뜨면 끝)

Windows 기본 이벤트 루프(Proactor)는 접속이 accept 도중 끊기면 WinError 64 를 내고
그 뒤로 접속을 못 받는 경우가 있어서, Selector 루프로 실행합니다.
Linux(EC2)에서는 이 파일 없이 uvicorn 을 그대로 써도 됩니다.
"""
import asyncio
import os
import sys

import uvicorn

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))


async def _serve_with_wakeup(server):
    """Windows Selector 루프는 Ctrl+C 를 다음 이벤트 때까지 처리하지 못함 → 0.5초마다 깨워서 바로 종료되게"""
    async def wakeup():
        while not server.should_exit:
            await asyncio.sleep(0.5)
    task = asyncio.ensure_future(wakeup())
    try:
        await server.serve()
    finally:
        task.cancel()


def main():
    config = uvicorn.Config("spoiler_server:app", host=HOST, port=PORT, workers=1)
    server = uvicorn.Server(config)
    try:
        if sys.platform == "win32":
            loop = asyncio.SelectorEventLoop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_serve_with_wakeup(server))
            finally:
                loop.close()
        else:
            asyncio.run(server.serve())
    except KeyboardInterrupt:
        # uvicorn 은 정상 종료를 마친 뒤 Ctrl+C 를 다시 던짐 → 오류 아님
        pass
    print("서버 종료")
    # Gemini 호출·분석 작업용 스레드가 남아 있으면 파이썬이 끝나지 않고 기다림 → 바로 종료
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()