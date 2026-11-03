# 영상 + 자막 사건 추출 실험

- 영상: `tears_of_steel_720p.mp4` / 자막: `tears_of_steel_en.srt` / 모델: `gemini-3.5-flash-lite`
- 자막 76줄, 청크 9개, 배치 2/2
- 사건 8건: 대사+화면 7, 대사만 0, **화면만 1**, 버림 0
- 이번 실행 토큰: 입력 0 / 출력 0 (0회 호출)

## 사건 목록 (시간순)

| 시각 | 근거 | 사건 | 화면 근거 | 대사 근거 |
|---|---|---|---|---|
| 00:23 | BOTH | 실험실에서 연구원들이 뇌 모형과 컴퓨터를 통해 로봇의 기억을 싱크하고 조작한다. | 01:11 연구원이 기계 장치와 컴퓨터 조작부를 다루고 있다. | "Robot's memory synced and locked!" |
| 00:53 | VISUAL | Celia가 로봇 손으로 Thom의 얼굴을 잡고 위협하다가 멈춘다. | 00:53 Celia가 로봇 손을 들어 Thom의 얼굴 앞에서 위협하듯 가리킨다. | - |
| 03:11 | BOTH | Barley가 무전기를 들고 남성에게 준비되었다며 지시한다. | 03:37 Barley가 무전기를 들고 다른 남성의 어깨를 잡고 있다. | "Are you ready? Of course you're ready, you're a rockstar." |
| 04:52 | BOTH | Thom과 Celia가 과거 로봇 손과 관련된 일로 다투는 연기를 한다. | 05:48 로봇 손을 가진 여자와 남자가 다투는 듯한 모습 | "Why don't you just admit that you're freaked out by my robot hand?" |
| 04:52 | BOTH | Thom이 기억 덮어쓰기(Memory overwrite) 작업을 진행하고 조정한다. | 04:57 Thom이 장비를 조작하며 모니터에 메모리 오버라이드 진행 바가 표시됨 | "Memory overwrite in progress!" |
| 05:56 | BOTH | Celia가 로봇 손으로 Thom의 얼굴을 움켜쥐고 위협한다. | 06:30 Celia의 로봇 손이 Thom의 입 주변을 강하게 잡고 있다. | "You broke my heart. I know. You forgot me on earth." |
| 06:27 | BOTH | 세트장에서 로봇과 인물들 간의 전투 촬영 및 폭발 상황이 발생한다. | 07:01 실내 세트장에서 거대한 폭발이 일어남 | "Quiet on set!" |
| 08:13 | BOTH | Thom이 기억 덮어쓰기가 완료되었다고 알린다. | 08:58 모니터에 덮어쓰기 완료 메시지가 뜸 | "Memory overwrite complete!" |

## 비교: 자막만으로 추출한 사건 (4건)

- Celia가 로봇 손 때문에 악마를 꾸며 거대하고 치명적인 로봇 발톱에 쫓기는 꿈을 꾼다고 톰에게 고백한다.
- 로봇의 메모리가 동기화되고 잠금 처리된다.
- 톰이 세상이 파괴될 이유가 없다고 말한다.
- 톰과 첼리아가 연기 연습 중 기억 덮어쓰기를 진행한다.

> 화면만(VISUAL) 사건 중 위 목록에 없는 것이 '영상을 더해서 새로 잡은 사건'입니다.
> 관리자가 보고서의 시각으로 영상을 직접 확인해 맞는지 판단하세요.
