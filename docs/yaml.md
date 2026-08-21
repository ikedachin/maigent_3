# YAML Configuration Documentation

このドキュメントは `.maigent/config.yaml` の設定項目を説明します。

設定ファイルは `config.toml` でも書けますが、このドキュメントではYAML形式を前提にします。

## 読み込み順

設定は以下の順で読み込まれます。後から読まれたものが前の値を上書きします。

1. `~/.maigent/`
2. アプリ直下の `.maigent/`
3. プロジェクト内の `.maigent/`

読み込み処理は `agent/config.py` の `load_runtime_config()` です。

注意:
- YAMLローダーは簡易実装です
- 基本的なネスト、真偽値、整数、リストに対応します
- 複雑なYAML機能が必要なら `config.toml` を推奨します
- 同じキーを複数回書くと後勝ちになり、前の定義が消える場合があります

## 全体例

```yaml
providers:
  openai:
    enabled: true
    model: gpt-5
    api_key: sk-...
    base_url: https://api.openai.com/v1
    api_mode: chat

llm:
  max_retries: 1

logging:
  llm_tail_chars: 100

final_evaluation:
  enabled: true
  max_retries: 3
  llm_max_retries: 1
  reasoning_effort: none
  max_output_tokens: 8192

tools:
  rag:
    enabled: true
  file_batch:
    enabled: true
  web_search:
    enabled: false
  sandbox:
    enabled: true
    image: maigent-sandbox:py311
    timeout_seconds: 20
    memory_limit_mb: 512
    pids_limit: 128
    cpus: 1
    install_libraries_on_run: false
    allowed_libraries:
      - pandas
      - numpy

tool_selector:
  enabled: true
  reasoning_effort: none
  max_output_tokens: 160
  max_retries: 1

initial_clarifier:
  enabled: true
  reasoning_effort: none
  max_output_tokens: 8192
  llm_max_retries: 1

dynamic_replanner:
  enabled: true
  reasoning_effort: true
  max_output_tokens: 8192
  llm_max_retries: 1

dynamic_finalizer:
  enabled: true
  reasoning_effort: none
  max_output_tokens: 8192
  llm_max_retries: 1

sandbox_code_generation:
  enabled: true
  reasoning_effort: low
  max_output_tokens: 32768
  llm_max_retries: 1
```

## providers

LLMプロバイダ設定です。

対応プロバイダ:
- `openai`
- `openai_compatible`
- `ollama`
- `lmstudio`
- `openrouter`
- `azure`
- `bedrock`

複数が `enabled: true` の場合は以下の順で最初に有効なものが使われます。

```text
openai -> openai_compatible -> ollama -> lmstudio -> openrouter -> azure -> bedrock
```

### providers.openai

```yaml
providers:
  openai:
    enabled: true
    model: gpt-5
    api_key: sk-...
    base_url: https://api.openai.com/v1
    api_mode: chat
```

項目:
- `enabled`: このプロバイダを使うか
- `model`: 使用モデル
- `api_key`: APIキー。環境変数 `OPENAI_API_KEY` でも可
- `base_url`: OpenAI互換APIのURL。環境変数 `OPENAI_BASE_URL` でも可
- `api_mode`: `auto`、`responses`、`chat`

`api_mode` の既定値は、すべてのプロバイダで `chat` です。

`api_mode`:
- `auto`: Responses APIを試し、失敗時にChat Completionsへフォールバック
- `responses`: Responses APIを使う
- `chat`: Chat Completions APIを使う

### providers.openai_compatible

vLLMやSGLangなど、OpenAI互換APIを提供する任意の推論サーバーに接続します。

```yaml
providers:
  openai_compatible:
    enabled: true
    model: Qwen/Qwen3.8-27B
    base_url: http://localhost:8000/v1
    api_mode: chat
    request:
      reasoning_effort: xhigh
      stream_options:
        include_usage: true
      extra_body:
        chat_template_kwargs:
          enable_thinking: true
          preserve_thinking: true
```

- `base_url` は必須です。
- `api_key` は任意です。空の場合は内部的にダミー値を使います。
- モデル名は推論サーバーが公開するIDをそのまま指定します。

### OpenAI互換プロバイダのrequest

`openai`、`openai_compatible`、`ollama`、`lmstudio`、`openrouter` は、プロバイダ配下の `request` をOpenAI SDKのリクエストへ渡します。`extra_body` はモデルや推論サーバー固有のパラメータに利用できます。

`model`、`messages`、`input`、`instructions`、`stream` は予約項目で、`request` に指定してもMaigent側の値が優先されます。`stream_options` はストリーミングChat時だけ送信され、非ストリーミングChatとResponses APIでは除外されます。

値はコード既定値、`request`、呼び出し固有値の順でマージされます。呼び出し固有の出力トークン数、`temperature`、`reasoning_effort` が最優先です。Chatで呼び出し固有値またはYAML値が `reasoning_effort: none` の場合、この項目はAPIへ送信されません。

## Logging configuration

LLMプロンプトやLLM応答をデバッグログへ出すときの表示量を設定できます。

```yaml
logging:
  llm_tail_chars: 100
```

項目:
- `llm_tail_chars`: ログへ表示する末尾文字数。標準は `100`

全文表示する場合:

```yaml
logging:
  llm_tail_chars: full
```

`full` の代わりに `all`、`unlimited`、または `0` 以下の数値も使えます。

### providers.ollama

```yaml
providers:
  ollama:
    enabled: false
    model: llama3.1
    base_url: http://localhost:11434/v1
    api_mode: chat
```

OllamaのOpenAI互換エンドポイントを使います。APIキーが空でも内部的にダミー値を使います。

### providers.lmstudio

```yaml
providers:
  lmstudio:
    enabled: false
    model: local-model
    base_url: http://localhost:1234/v1
    api_mode: chat
```

LM StudioのOpenAI互換エンドポイントを使います。

### providers.openrouter

```yaml
providers:
  openrouter:
    enabled: false
    model: openai/gpt-4o-mini
    api_key: sk-or-...
    base_url: https://openrouter.ai/api/v1
    api_mode: chat
    http_referer: http://localhost:8000
    x_title: Maigent Agent Web
```

追加ヘッダ:
- `http_referer`: `HTTP-Referer`
- `x_title`: `X-Title`

### providers.azure

```yaml
providers:
  azure:
    enabled: false
    model: azure-deployment-name
    api_key: azure-key
    azure_endpoint: https://your-resource.openai.azure.com
    api_version: 2024-02-15-preview
    api_mode: chat
```

Azure OpenAIを使います。

必須:
- `model`: Azureのdeployment名
- `api_key`
- `azure_endpoint`

### providers.bedrock

```yaml
providers:
  bedrock:
    enabled: false
    model: anthropic.claude-3-5-sonnet-20240620-v1:0
    region: ap-northeast-1
    profile: default
```

AWS Bedrockを使います。

認証:
- `profile` を使う
- または `aws_access_key_id`、`aws_secret_access_key`、`aws_session_token` を設定する
- 環境変数も利用可能

## トップレベル model/api_key/base_url

`providers` を使わない場合、トップレベルの以下をOpenAI互換APIとして扱います。

```yaml
model: gpt-5
api_key: sk-...
base_url: https://api.openai.com/v1
api_mode: chat
```

`providers` がある場合は、基本的に有効なproviderの設定が優先されます。

## llm

内部LLM補助呼び出しの共通設定です。

```yaml
llm:
  max_retries: 1
```

### llm.max_retries

`None`、空文字、例外が返った場合に、同じLLM補助呼び出しを何回リトライするかです。

例:
- `0`: リトライなし。初回のみ
- `1`: 初回 + 1回リトライ
- `2`: 初回 + 2回リトライ

上限は実装上5です。

対象:
- `initial_clarifier`
- `dynamic_replanner`
- `dynamic_finalizer`
- `final_evaluation`
- `rag_decision`
- `rag_candidate_judge`
- `sandbox_code_generation`

個別セクションの `llm_max_retries` がある場合は、そちらが優先されます。

## final_evaluation

回答をユーザーへ返す前の最終評価設定です。

```yaml
final_evaluation:
  enabled: true
  max_retries: 3
  llm_max_retries: 1
  reasoning_effort: none
  max_output_tokens: 8192
```

項目:
- `enabled`: 最終評価を行うか
- `max_retries`: 評価NG時に別プランで再実行する回数
- `llm_max_retries`: 評価LLMが空応答/None/例外を返したときの短いリトライ回数
- `reasoning_effort`: Responses APIに渡すreasoning effort
- `max_output_tokens`: 評価LLMの最大出力トークン数。未指定時は `8192`、指定時はその値を使用します

注意:
- `max_retries` は0から3に丸められます
- UIから保存した有効/無効と `max_retries` はDBの `AppSetting` が優先されます
- `llm_max_retries` は評価NG時の再プラン回数ではありません

## tools

実行ツールの有効/無効と詳細設定です。

### tools.rag

```yaml
tools:
  rag:
    enabled: true
```

許可済みローカルファイル検索を有効化します。

`enabled: true` の場合でも、読み取り許可パスがなければ検索対象はありません。

### tools.file_batch

```yaml
tools:
  file_batch:
    enabled: true
    max_output_tokens: 4096
    max_retries: 0
    reasoning_effort: none
```

許可済みローカルファイルをフォルダ横断でmap-reduce処理する設定です。全ファイル要約や一覧化・横断分析など、単一クエリのBM25検索では扱いにくいタスクで `rag` の代わりに選ばれます。

項目:
- `enabled`: file_batch実行を有効化するか。`tools.file_batch` が未定義でも `tools.rag.enabled` が `true` なら自動的に有効になります(明示的に `false` を指定すれば無効化できます)
- `max_output_tokens`: map段階の要約LLM呼び出しの最大出力トークン数。未指定時は `4096`
- `max_retries`: map段階のLLMが空応答/None/例外を返した場合の再試行回数
- `reasoning_effort`: map段階のreasoning effort

### tools.web_search

```yaml
tools:
  web_search:
    enabled: false
    api_key: tvly-...
    max_results: 5
    timeout_seconds: 10
```

外部Web検索の設定です。[Tavily](https://tavily.com/) の検索APIを使います。

項目:
- `enabled`: web_search実行を有効化するか
- `api_key`: Tavily APIキー。環境変数 `TAVILY_API_KEY` でも可
- `max_results`: 取得する検索結果件数。1から10に丸められます。既定は5
- `timeout_seconds`: HTTPタイムアウト秒。1から30に丸められます。既定は10

`api_key` が未設定の場合、計画に選択されても「APIキーが未設定です」という明確なメッセージを返すだけで、無言で成功したことにはしません。

### tools.sandbox

```yaml
tools:
  sandbox:
    enabled: true
    image: maigent-sandbox:py311
    timeout_seconds: 20
    memory_limit_mb: 512
    pids_limit: 128
    cpus: 1
    install_libraries_on_run: false
    allowed_libraries:
      - pandas
      - numpy
```

項目:
- `enabled`: sandbox実行を有効化するか
- `image`: Dockerイメージ名
- `timeout_seconds`: 実行タイムアウト秒。1から600に丸められます
- `memory_limit_mb`: コンテナのメモリ上限(MB)。64から8192に丸められます。既定は512
- `pids_limit`: コンテナ内で生成できるプロセス/スレッド数の上限。16から2048に丸められます。既定は128
- `cpus`: コンテナに割り当てるCPUコア数。0.1から8.0に丸められます。既定は1
- `install_libraries_on_run`: 実行ごとに `allowed_libraries` をpip installするか
- `allowed_libraries`: 実行時インストールを許可するライブラリ

推奨:
- 通常は `install_libraries_on_run: false`
- 必要ライブラリはDockerイメージに事前インストールする

sandboxの安全性:
- 入力ファイルは直接マウントしません
- 通常はDockerネットワークを無効化します
- `docker run` に `--memory` / `--pids-limit` / `--cpus` を付与し、暴走コードやフォーク爆弾がホストへ影響しないよう制限します
- ファイル保存は typed `maigent_sandbox_result` JSONをstdoutへ出し、ホスト側brokerが権限検証して行います

## tool_selector

LLMに初期ツール列を選ばせる設定です。

```yaml
tool_selector:
  enabled: true
  reasoning_effort: none
  max_output_tokens: 160
  max_retries: 1
```

項目:
- `enabled`: LLMツール選択を有効化するか
- `reasoning_effort`: reasoning effort
- `max_output_tokens`: 最大出力トークン数
- `max_retries`: 空応答/不正JSON/無効JSON時に選択処理全体を再試行する回数

注意:
- `max_retries` はtool selector専用の既存設定です
- tool selector は `llm.max_retries` や `llm_max_retries` ではなく、この `max_retries` で再試行します

## initial_clarifier

初期プラン前に、不足情報があるかをLLMで判定する設定です。

```yaml
initial_clarifier:
  enabled: true
  reasoning_effort: none
  max_output_tokens: 8192
  llm_max_retries: 1
```

項目:
- `enabled`: 初期確認判定を有効化するか
- `reasoning_effort`: reasoning effort
- `max_output_tokens`: 最大出力トークン数。未指定時は `8192`、指定時はその値を使用します
- `llm_max_retries`: 空応答/None/例外時の再試行回数

質問が必要と判定された場合は、理由と最大3問の確認事項を返してエージェント実行を止めます。

## dynamic_replanner

1タスク実行後の動的リプラン設定です。

```yaml
dynamic_replanner:
  enabled: true
  reasoning_effort: true
  max_output_tokens: 8192
  llm_max_retries: 1
```

項目:
- `enabled`: 動的リプランを有効化するか
- `reasoning_effort`: reasoning effort。`true` は `medium`、`false` は `none` として扱います
- `max_output_tokens`: 最大出力トークン数。未指定時は `8192`、指定時はその値を使用します
- `llm_max_retries`: 空応答/None/例外時の再試行回数

返せるアクション:
- `keep`
- `replace`
- `finish`

## dynamic_finalizer

最終成果物の扱いをLLMで判断する設定です。

```yaml
dynamic_finalizer:
  enabled: true
  reasoning_effort: none
  max_output_tokens: 8192
  llm_max_retries: 1
```

項目:
- `enabled`: 最終ルーティングを有効化するか
- `reasoning_effort`: reasoning effort
- `max_output_tokens`: 最大出力トークン数。未指定時は `8192`、指定時はその値を使用します
- `llm_max_retries`: 空応答/None/例外時の再試行回数

返せるアクション:
- `save`: 保存相当で終了
- `discard`: 一時検証として終了
- `add_tasks`: 追加タスクを実行

## sandbox_code_generation

sandboxで使うPythonコードをLLM生成する場合の設定です。

```yaml
sandbox_code_generation:
  enabled: true
  reasoning_effort: low
  max_output_tokens: 32768
  llm_max_retries: 1
```

項目:
- `enabled`: LLMによるPythonコード生成を有効化するか。未指定時は `true`
- `reasoning_effort`: コード生成時のreasoning effort。`xhigh` も指定可能
- `max_output_tokens`: コード生成LLMの最大出力トークン数。未指定時は `32768`
- `llm_max_retries`: コード生成LLMが空応答/None/例外を返した場合の再試行回数

## rag_decision

必要な場合だけ追加できます。

```yaml
rag_decision:
  llm_max_retries: 1
```

ローカルファイル検索が必要かをLLMで判定する補助呼び出しのリトライ回数です。

## rag_candidate_judge

必要な場合だけ追加できます。

```yaml
rag_candidate_judge:
  llm_max_retries: 1
```

BM25で弱い候補をLLMで関連判定する補助呼び出しのリトライ回数です。

## reasoning_effort

設定可能な値:

- `none`
- `minimal`
- `low`
- `medium`
- `high`
- `xhigh`（`sandbox_code_generation` とプロバイダの `request` で指定可能）
- `true`
- `false`

`true` は `medium` として扱います。`false` は `none` として扱います。

注意:
- Chat Completions APIでは `none` 以外を `reasoning_effort` として送信し、`none` は項目自体を省略します
- Responses APIでは `reasoning: {"effort": ...}` として送信されます
- プロバイダや互換APIによって対応状況が異なります

## max_output_tokens

内部判断用LLM呼び出しの最大出力トークン数です。

用途:
- ツール選択や評価などの短いJSON出力を低コスト・低遅延にする
- 長すぎる説明を抑制する

注意:
- 小さすぎるとJSONが途中で切れてパースできない場合があります
- `dynamic_replanner` で長いタスク列を返させる場合は、未指定時の `160` から大きめに増やしてください

## 設定変更時の確認コマンド

現在読み込まれている設定を確認する例:

```bash
uv run python manage.py shell -c "from agent.config import load_runtime_config; c=load_runtime_config(''); print(c.active_provider, c.model); print(c.tool_enabled('sandbox')); print(c.control_config('dynamic_finalizer'))"
```

テスト:

```bash
uv run python manage.py test agent
```
