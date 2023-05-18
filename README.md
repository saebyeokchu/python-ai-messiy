## Meissy 🛫

Meissy는 0.01버전으로 시그모이드 함수를 사용하여 NER(Name Entity Recognition)을 수행합니다.

아래와 같이 실행할 수 있습니다.

Requirements

```css
- re
- soyspacing
- tensorflow
```

입력

```css
python3.9 messiy.py "이 표에서 장철수의 전화번호를 찾아줘"
```

출력

```css
[
	{ 
		name : "장철수"
		entity : "name"
	}
]
```

현재 버전은 임시 구현이고 추후 LSTM과 KoBERT를 활용한 확장을 진행할 예정입니다

문의 : cuu2252@gmail.com
