# Multilingual Safety Head


```
models/ <-llama等のモデルの格納場所
```
[Todo List](./ToDo.md)

## 実験設定
- Generalized Shipsの方
- 各言語ごとにオーダー100件
- Figure4.5の関係は？
- 結局どのheadを削除してるの？
- shortだと2時間 しかない…
- モデルはllama3 ⇒(やってみて言語性能が不安ならllama2)

モデルサイズの違いによって，内部挙動が違うケースがある．
⇒複数のモデルでheadの分布の違いを見る．


### 使用するデータ
##### 使用する言語
* 主要言語
英語
* 対応言語
ドイツ語 ヒンディー語
* 非公式対応言語
日本語
##### 使用するベンチマークデータ
- unsafe instruction topivc
- crime and ilegal activity
- insult
* 量はそれぞれ100件

### モデル
##### 使用するモデル
- llaam3.2-3B
- llama3.1-8B
- llama3.2-14B
- llama3.1-70B
いずれ405Bもやりたい
##### 量子化について
- bf16でいい