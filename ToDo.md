# To-Do List

## 動作確認
<<<<<<< HEAD
- [×] ヘッドを調整して保存
- [×] 調整済みのモデルで推論
- [×] ships  
  - Error : Out of Memory
- [×] 改良ships  
  - Error : Out of Memory
- [×] wisteria触る
- [×] ships (3B)
- [×] 改良ships (3B)

## multilingual safety headの実験準備
- [×] 使うモデル決める
- [×] データセットの決定 
- [×] データセットの変換
- [×] バッチ処理化 
=======
- [x] ヘッドを調整して保存
- [x] 調整済みのモデルで推論
- [x] ships  
  - Error : Out of Memory
- [x] 改良ships  
  - Error : Out of Memory
- [x] wisteria触る
- [x] ships (3B)
- [x] 改良ships (3B)

## multilingual safety headの実験準備
- [x] 使うモデル決める
- [x] データセットの決定 
- [x] データセットの変換
- [ ] バッチ処理化 
>>>>>>> f9d86215a622943e1ec7841dd63e707fd57108b3
  - 1回のsearch_stepでLayer*Headのタスクが存在．DatapParallelは向いてない   
  - タスク並列でヘッド毎の推論を複数GPUに割り振る ⇒ fp16なら20Bくらいまではこの方法で行けるはず？
    - (70Bやるときはまた別途分散学習フレームワーク&フレームワーク並列で実装)

### ヘッドの特定
- llama3.2-3B-Instructでheadの特定
<<<<<<< HEAD
  - [×] 英語
  - [×] ドイツ語
  - [×] ヒンディー語
  - [×] 日本語
  - [×] ヘッドの分布可視化
- llama3.1-8B-Instructでheadの特定
  - [×] 英語
  - [×] ドイツ語
  - [×] ヒンディー語
  - [×] 日本語
  - [×] ヘッドの分布可視化
- llama2-7B-Instructでheadの特定
  - [×] 英語
  - [×] ドイツ語
  - [×] ヒンディー語
  - [×] 日本語
  - [×] ヘッドの分布可視化
- [×] search config - 2の場合

### Saharaがあってるの確認
- squadデータで検証
  - [ ] llama3.1-8B-Instruct
  - [ ] llama3.2-3B-Instruct

### Saharaのバグ修正
- [ ] maskされてない & ROPEがおかしい
- [ ] ヘッドの特定
  - [ ] 8B
    - [ ] ja en de hi multilingual
    - [ ] squad
  - [ ] 3B
    - [ ] ja en de hi multilingual
    - [ ] squad

### ASR ベースの実装
- [ ] NGで出てくるものリスト作成
- [ ] レイヤ毎にマスキングしてASRをルールベースで割り出すコードの作成
- [ ] 検証(サンプルは100)



=======
  - [ ] 英語
  - [ ] ドイツ語
  - [ ] ヒンディー語
  - [ ] 日本語
  - [ ] ヘッドの分布可視化
- llama3.1-8B-Instructでheadの特定
  - [ ] 英語
  - [ ] ドイツ語
  - [ ] ヒンディー語
  - [ ] 日本語
  - [ ] ヘッドの分布可視化
- llama3.2-14B-Instructでheadの特定
  - [ ] 英語
  - [ ] ドイツ語
  - [ ] ヒンディー語
  - [ ] 日本語
  - [ ] ヘッドの分布可視化
- llama3.1-70B-Instructでheadの特定
  - [ ] 英語
  - [ ] ドイツ語
  - [ ] ヒンディー語
  - [ ] 日本語
  - [ ] ヘッドの分布可視化
>>>>>>> f9d86215a622943e1ec7841dd63e707fd57108b3

### その他
- [ ] ベースラインのASRの実験(調整なし)
- [ ] 調整したモデルでASR実験
