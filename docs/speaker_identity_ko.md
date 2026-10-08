# 화자 이름 표시 사용법

Sortformer 화자 분리 결과(화자 1~4)를 미리 등록한 사람 이름으로 바꿔 표시하는 기능입니다.

- 회의 중 같은 화자로 분류된 발화에서 확실한 구간만 골라 TitaNet 임베딩을 만들고, 등록된 사람들과 비교합니다.
- 누적 발화 10초 이상, 1위 점수 0.70 초과, 1위와 2위 차이 0.10 초과를 모두 만족하면 이름을 표시합니다. 하나라도 만족하지 않으면 번호를 그대로 표시합니다.
- 등록한 사람들을 **profile 그룹**(예: `OOD-VIL`, `회사`)으로 나눠 두고, 웹 화면에서 녹음 전에 그룹을 고를 수 있습니다.

## 1. 설치 (맥, Apple Silicon)

```bash
brew install ffmpeg
git clone -b persistent-speaker-identity https://github.com/guhyun9454/WhisperLiveKit
cd WhisperLiveKit
python3.11 -m venv .venv          # NeMo는 Python 3.10~3.12 필요
source .venv/bin/activate
pip install -e ".[mlx-whisper,diarization-sortformer,cpu]"
```

- 전사(Whisper)는 mlx-whisper 백엔드를 사용하므로 맥 GPU에서 실행됩니다.
- 화자 분리(Sortformer)와 이름 판정(TitaNet)은 CUDA가 없으면 CPU에서 실행됩니다.
- 처음 실행할 때 Sortformer, TitaNet, Whisper 모델을 Hugging Face에서 내려받습니다.

## 2. profile 준비

`--speaker-profiles`로 지정한 폴더 아래의 **하위 폴더 하나가 그룹 하나**입니다. 파일 하나가 사람 한 명입니다.

```
wlk-profiles/
├── OOD-VIL/
│   ├── 권구현.json
│   ├── 박경문_교수님.json
│   └── ...
└── 회사/
    ├── 권구현.json
    ├── 다인님.json
    └── ...
```

한 사람이 여러 그룹에 포함돼도 됩니다(파일을 각 폴더에 복사).

**새 사람 등록:** 그 사람만 말하는 녹음 파일(30초 이상 권장)로 만듭니다. 같은 이름으로 다시 실행하면 기존 profile에 추가됩니다.

```bash
python -m whisperlivekit.diarization.speaker_identity enroll wlk-profiles/회사 다인님 dain1.wav dain2.wav
```

wav, flac 파일을 사용합니다. m4a는 먼저 변환합니다: `ffmpeg -i dain.m4a -ac 1 -ar 16000 dain.wav`

profile 파일에는 목소리 임베딩이 포함됩니다. git 저장소에 올리지 마세요.

## 3. 서버 실행

```bash
wlk --model base --lan ko \
    --diarization --diarization-backend sortformer \
    --speaker-profiles ~/wlk-profiles
```

브라우저에서 `http://localhost:8000`을 엽니다.

## 4. 웹에서 그룹 선택

1. 녹음 버튼 옆 설정(톱니바퀴) 버튼을 누릅니다.
2. **Speaker profiles** 목록에서 그룹을 고릅니다. 괄호 안 숫자는 등록 인원이고, 항목 위에 마우스를 올리면 이름 목록이 표시됩니다.
3. 녹음을 시작합니다. 선택한 그룹은 다음 녹음부터 적용되고, 브라우저에 저장됩니다.

`None`을 고르면 이름을 표시하지 않고 원래처럼 번호만 표시합니다.

웹 화면 없이 연결할 때는 WebSocket URL에 그룹을 지정합니다: `ws://localhost:8000/asr?speaker_group=회사`. 그룹 목록은 `GET /speaker-groups`로 조회합니다.

## 5. 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--speaker-profiles DIR` | 없음 | profile 폴더. 지정하면 기능이 켜집니다 |
| `--speaker-threshold` | 0.70 | 1위 점수 기준. 높이면 오답이 줄고 Unknown이 늘어납니다 |
| `--speaker-margin` | 0.10 | 1위와 2위 점수 차이 기준 |
| `--speaker-candidates "A,B"` | 없음 | 선택한 그룹 안에서 이 사람들만 비교합니다 |

결과 JSON의 각 line에는 `speaker_name`(판정 전에는 `null`)과 `speaker_confidence`가 추가됩니다.

## 6. 참고 사항

- **맥 실행 속도는 아직 확인하지 않았습니다.** 실시간 검증은 CUDA GPU(RTX A2000)에서만 했습니다. Sortformer를 CPU로 실행하면 실시간보다 느릴 수 있습니다. 화면 상단의 diarization lag 값이 계속 커지면 CPU 속도가 부족한 상태입니다.
- Sortformer는 화자를 최대 4명까지만 구분합니다. 5명 이상 회의에서는 두 사람이 한 화자로 합쳐질 수 있고, 그러면 한 사람 이름이 다른 사람 발화에도 표시됩니다.
- 이름은 화자 번호 단위로 지정됩니다. 판정 전 발화는 번호로 표시되고, 판정 후에는 그 화자의 이전 발화에도 이름이 표시됩니다.
