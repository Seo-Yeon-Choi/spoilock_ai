# 파일 정리 내역

입력: `spoilock_subtitle_starter(2).zip`
원본은 변경하지 않았으며 새 프로젝트를 만들었습니다. 기존 코드·실험 결과는 `archive/legacy_reference.zip`에 참고용으로 보관했습니다. 이 ZIP은 과거 파일 일부를 모은 참고본이며 독립 실행 프로젝트가 아닙니다. 원본 전체는 사용자가 올린 원본 ZIP입니다.

| 기존 | 정리 후 | 이유 |
|---|---|---|
| `spoiler_server.py` | 같은 이름 | 유일한 서버 진입점 유지 |
| `spoiler_events_multi.py` | `event_store.py` + `tools/events.py` | 저장·검색·검증만 공통 모듈로 남기고 중복 main, 단일 회차 CLI, 출력 가로채기 제거 |
| `check_spoiler.py` | 연결 함수만 `gemini_client.py`로 추출, 원본은 보관 | 서버는 get_client만 사용. 과거 청크 검색·번역·RRF 로직은 현재 사건 서버에서 호출하지 않음 |
| `prefilter.py` | 같은 이름 | 기존 필터 판단 유지, 키워드 파일 경로만 변경 |
| `prefilter_keywords.txt` | `config/prefilter_keywords.txt` | 설정 파일 분리 |
| `chat_tester.html` | `web/chat_tester.html` | 기존 UI와 `/` 경로 유지 |
| `pipeline.py` | 전처리만 `tools/preprocess.py`, 원본은 보관 | 사건 서버가 쓰지 않는 과거 청크 컬렉션 인덱싱·검색을 실행 경로에서 제거 |
| `extract_events.py` / `spoiler_events_demo.py` | `tools/extract_events.py` 하나로 통합 | 이름과 달리 두 파일 모두 사건 추출기. demo 쪽의 실패 묶음 기록·건너뛰기 기능을 보존 |
| `try_server.py` | `tools/request_judge.py` | 요청 클라이언트임을 명확히 함 |
| `run_tests_30.py` | `tests/run_cases.py` | 문장마다 CLI 프로세스를 시작하지 않고 `/judge` 요청으로 실제 서버 로직 테스트 |
| `test_cases_30.json`, `.csv`, `prefilter_cases.csv` | `tests/fixtures/` | 기존 라벨 그대로 보존 |
| `calibrate_prefilter.py`, `compare_configs.py` | `tools/` | 현재 서버에 유용한 선택 점검 도구, 이동한 경로 반영 |
| `timing_check.py` | 보관 | 기존 CLI를 monkey-patch하는 도구. 서버 timings_ms와 요청 클라이언트로 확인 가능 |
| `gemini_ping.py` | 보관 | 개별 연결 실험. 필요한 경우 원본 참고 가능 |
| `verify_chroma.py` | 보관 | 사건 컬렉션이 아닌 예전 청크 컬렉션을 검사. 새 도구의 list와 /health 사용 |
| `README.md`, `README_GEMINI.md` | 현재 README 하나로 통합, 원본 보관 | 기존 명령·새 명령 혼동 방지 |
| `requirements.txt`, `requirements-gemini.txt` | `requirements.txt` 하나로 통합 | FastAPI·uvicorn 누락도 포함 |
| `gitignore-additions.txt` | 실제 `.gitignore`, 원본 보관 | 환경·DB·캐시 제외 |
| `event_extraction_prompt.txt` | 보관 | 코드에서 읽지 않는 참고 문서 |
| `search_result.txt`, 기존 `data/tests`, `data/events/debug`, 기존 캐시 | 보관 | 과거 실행 산출물과 현재 실행 데이터 구분 |
| `data/raw`, `data/processed`, 사건 초안 6개 | 원래 경로 그대로 | 재인덱싱·원문 검증에 필요한 데이터 보존 |

## 구현 변경 범위

1. 서버 초기화에서 함수 전역 재할당을 없애고 이미 로딩한 모델·컬렉션을 검색 함수 인자로 전달합니다.
2. 임베딩 모델을 프로세스 내에서 재사용하므로 여섯 회차 인덱싱에서도 같은 모델 객체를 재사용합니다.
3. API URL, 요청/응답 필드, LLM 프롬프트, 캐시 키, 1차 필터와 회차 노출 판정은 유지합니다.
4. 서버 옵션을 루트 `.env`에서 읽습니다. 첫 실행 다운로드가 막히지 않도록 HF offline 기본값만 1에서 0으로 바꿉니다.
5. 테스트 30개는 기존 라벨을 유지하고 API 필드 `required_episode_number`를 비교합니다. 기존 테스트와 맞추기 위해 1차 필터는 off로 호출합니다.
6. 보정 도구의 '누출 0' 문구를 해당 테스트 표본의 결과임이 드러나도록 수정했습니다.

## 이전 폴더 처리

새 폴더에서 정상 실행을 확인할 때까지 기존 폴더는 백업으로 둡니다. 로컬에만 있는 `.env`, `chroma_data`는 필요에 따라 새 폴더로 복사합니다. 이 작업은 사용자 Windows 파일을 직접 삭제하거나 이동하지 않았습니다.
