# To-Do List

## 動作確認
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
  - 1回のsearch_stepでLayer*Headのタスクが存在．DatapParallelは向いてない   
  - タスク並列でヘッド毎の推論を複数GPUに割り振る ⇒ fp16なら20Bくらいまではこの方法で行けるはず？
    - (70Bやるときはまた別途分散学習フレームワーク&フレームワーク並列で実装)

### ヘッドの特定
- llama3.2-3B-Instructでheadの特定
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

### その他
- [ ] ベースラインのASRの実験(調整なし)
- [ ] 調整したモデルでASR実験
