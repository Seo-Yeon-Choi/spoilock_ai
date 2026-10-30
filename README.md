# SPOILOCK AI 서버 (FastAPI)

스포일러 사건 DB(Chroma)와 판정 API. Spring 이 이 서버를 호출한다.

| 기능 | 경로 |
|---|---|
| 채팅·게시글 판정 | `POST /judge` |
| 자막(+영상) 분석 작업 | `/admin/subtitles/jobs` |
| 영상 음성 인식 분석 | `/admin/videos` |
| 사건 조회·검수·수동 등록 | `/admin/events`, `/kb/episodes` |

## 실행 (Windows)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env   # 값 채우기
.venv\Scripts\python.exe -m uvicorn spoiler_server:app --host 0.0.0.0 --port 8000
```

## 사건 DB 칸 관리

```powershell
.venv\Scripts\python.exe kb_tool.py list
```

`chroma_data/`, `data/`, `.env` 는 커밋하지 않는다.
