# Maigent Agent Web

ローカルで動くDjango製のエージェントチャットWebアプリです。
## 🚧 UNDER CONSTRUCTION 🚀
### ✨ このリポジトリは現在進化中です。いっしょに育てていきましょう！

![](./maigent.png)

## Setup

```bash
uv sync
uv run python manage.py migrate
uv run python manage.py createsuperuser
docker build -t maigent-sandbox:py311 docker/sandbox
uv run python manage.py runserver
```

データベースの閲覧・編集は `http://127.0.0.1:8000/admin/` から行えます。初回だけ `createsuperuser` で管理者を作成し、そのアカウントでログインしてください。

設定は `config.toml` または `config.yaml` で管理します。読み込み順は `~/.maigent/`、アプリ直下の `.maigent/`、プロジェクト内の `.maigent/` です。後から読み込まれた設定が前の設定を上書きします。

```yaml
model: gpt-5
api_key: sk-...
base_url: https://api.openai.com/v1
api_mode: chat

llm:
  max_retries: 1

logging:
  llm_tail_chars: 100

final_evaluation:
  enabled: false
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

multi_agent:
  enabled: true
  max_workers: 3
  parallel_tools: true
  progress_visible: true

sandbox_code_generation:
  enabled: true
  reasoning_effort: low
  max_output_tokens: 32768
  llm_max_retries: 1
```

`providers` を使わない場合は、上記のトップレベル設定をOpenAI互換APIとして扱います。`model` または `default_model` は必須です。`OPENAI_MODEL` はモデル未設定時のみ補完に使われます。APIキーはDBへ保存しません。

`llm.max_retries` は、空応答・`None`・例外が返ったLLM補助呼び出しの共通リトライ回数です。各セクションの `llm_max_retries` で個別に上書きできます。

`logging.llm_tail_chars` は、LLMプロンプトやLLM応答をデバッグログへ出すときの末尾文字数です。標準は `100` です。全文を出す場合は `full`、`all`、`unlimited`、または `0` を指定します。

`final_evaluation.enabled` を `true` にすると、回答をブラウザへ返す前に最終評価を行い、不十分な場合は最初のプランから最大 `max_retries` 回まで再実行します。`max_retries` は 0 から 3 の範囲に丸められます。最終評価のLLM呼び出しは `reasoning_effort`、`max_output_tokens`、`llm_max_retries` で軽量化と応答失敗時の再試行を設定できます。画面から保存した最終評価の有効/無効とリトライ回数はDBの `AppSetting` に保存され、設定ファイル値より優先されます。ただし、プランが `rag` / `sandbox` / `web_search` / `file_batch` のいずれも使わず直接回答した場合は、`final_evaluation.enabled` の値によらず評価LLM呼び出し自体を行わず回答をそのまま返します。ツールを使わない雑談的な回答は検証対象を持たないため常にスキップし、ツールで外部情報や計算結果を取り込んだ回答だけを評価する設計です。

`initial_clarifier` と `tool_selector` を両方有効にしている場合、この2つのLLM判定は順番待ちさせず並列実行します（`ThreadPoolExecutor`で同時に投げて両方の結果を待ち合わせ）。DB参照を伴う下準備（利用可能ツール一覧の取得、ローカルコンテキストの補正）は並列化せずリクエストのスレッド上で行い、ネットワーク越しのLLM呼び出し部分だけを並列化しているため、応答内容は逐次実行時と変わらず、体感速度だけが改善します。さらに、`tool_selector` が `rag` / `sandbox` / `web_search` / `file_batch` のいずれかを含む具体的な実行プランを見つけた場合は、`initial_clarifier` が「確認が必要」と判定していてもその確認質問は表示せず、`tool_selector` が見つけた実行プランをそのまま採用します。ツールを使った具体的な実行方針が既に見つかっているなら、ユーザーに聞き返す必要はないという考え方です。`tool_selector` が実行可能なツールを見つけられなかった場合（直接回答のみ、または判定失敗）は、従来どおり `initial_clarifier` の判定に従います。

`multi_agent.enabled` は、1つのユーザー依頼の内部で複数worker agentを並列実行するかを制御します。標準は有効で、`max_workers` は 1 から 5 の範囲に丸められます。v1では同一スレッドへの複数ユーザー送信を同時処理するのではなく、1つのassistant回答の中で `research` / `compute` / `verify` workerを走らせ、最後に1つの回答へ統合します。`parallel_tools` が `true` の場合、RAGやsandboxなど独立可能なツールをworkerへ分割します。`file_batch` はフォルダ横断タスクをmap-reduceバッチへ分け、必要に応じて `max_workers` まで並列map処理します。`progress_visible` が `true` の場合、ブラウザの進捗欄にagent別の状態を表示します。

プラン作成時の評価基準文は `prompt/evaluation_criteria.txt` から読み込みます。コード側は依頼内容に応じて `base`、`rag`、`sandbox`、`summary`、`list`、`rag_selected` の各セクションを選び、文面はプロンプトファイル側で調整できます。

エージェント実行は、初期プランを `AgentState.plan_queue` に入れ、1タスクごとに実行履歴を保存しながら進めます。回答生成、sandboxコード生成、最終評価にはスレッド内の完了済み会話履歴をすべて入力し、最新ユーザーメッセージを明示します。プラン作成やRAG検索クエリの判定は最新ユーザーメッセージを基準にします。`initial_clarifier.enabled` が `true` の場合、初期プラン前にLLMが不足情報の有無をJSONで判定し、必要なら理由付きの確認質問を返して実行を止めます。`tool_selector.enabled` が `true` の場合、LLMが利用可能tools一覧から `rag` / `file_batch` / `sandbox` / `web_search` / `final` の実行順を短いJSONで選びます。この判定呼び出しは `reasoning_effort: none` と小さい `max_output_tokens` を指定して、低遅延・低コストに寄せています。空応答や不正JSONなどで失敗した場合は `max_retries` 回だけ再試行し、それでも失敗した場合は従来のルールベースプランへフォールバックします。`dynamic_replanner.enabled` が `true` の場合、各タスク後にLLMリプランナーが `goal`、`plan_history`、`task_history`、現在の `plan_queue` を見て、キュー維持・差し替え・終了をJSONで判断します。`dynamic_finalizer.enabled` が `true` の場合、終了時に成果物を保存相当で扱うか、破棄相当で終えるか、追加検証タスクを挿入するかをLLMで分岐します。設定未指定の `RuntimeConfig` ではこれらが有効です。

YAMLローダーはこのアプリ内の簡易実装です。基本的なネスト、真偽値、整数、リストには対応しますが、一般的なYAMLの全機能を使う設定は `config.toml` を推奨します。

## LLM provider configuration

`providers.<name>.enabled` で使用するAPIを切り替えられます。複数を `true` にした場合は `openai`, `openai_compatible`, `ollama`, `lmstudio`, `openrouter`, `azure`, `bedrock` の順で最初に有効なものを使用します。`providers` を書いたうえで全て `false` にした場合、active provider はなしになり、モデル未設定として扱われます。
ちなみに、作者が確認できているのはOpenAI互換APIのみです。

```yaml
providers:
  openai:
    enabled: true
    model: gpt-5
    api_key: sk-...
    api_mode: chat

  openai_compatible:
    enabled: false
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

  ollama:
    enabled: false
    model: llama3.1
    base_url: http://localhost:11434/v1
    api_mode: chat

  lmstudio:
    enabled: false
    model: local-model
    base_url: http://localhost:1234/v1
    api_mode: chat

  openrouter:
    enabled: false
    model: openai/gpt-4o-mini
    api_key: sk-or-...
    base_url: https://openrouter.ai/api/v1
    api_mode: chat
    http_referer: http://localhost:8000
    x_title: Maigent Agent Web

  azure:
    enabled: false
    model: azure-deployment-name
    api_key: azure-key-...
    azure_endpoint: https://your-resource.openai.azure.com
    api_version: 2024-02-15-preview
    api_mode: chat

  bedrock:
    enabled: false
    model: anthropic.claude-3-5-sonnet-20240620-v1:0
    region: ap-northeast-1
    profile: default
    # profileの代わりにキーを設定する場合:
    # aws_access_key_id: ...
    # aws_secret_access_key: ...
    # aws_session_token: ...
```

`openai_compatible` はvLLMやSGLangなどの汎用OpenAI互換エンドポイント向けで、`base_url` は必須、APIキーは任意です。OllamaとLM StudioもAPIキー未指定時はローカル用のダミーキーで接続します。OpenRouterは任意で `http_referer` と `x_title` をヘッダーへ付与できます。Azureは `AzureOpenAI` クライアント、AWS Bedrockは `boto3` の Converse API を使用します。

OpenAI互換プロバイダでは、`request` 配下の値をSDKリクエストへ渡せます。`extra_body` はQwenなどのモデル固有設定に利用できます。`model`、`messages`、`input`、`instructions`、`stream` はMaigentが管理するため、`request` に書いても上書きされません。`stream_options` はストリーミングChat時だけ使われます。マージ順はコード既定値、`request`、呼び出し固有値の順で、呼び出し固有のトークン数・温度・reasoning設定が優先されます。Chatの `reasoning_effort: none` はAPIへ送信しません。

環境変数も利用できます。OpenAIは `OPENAI_API_KEY` / `OPENAI_BASE_URL`、OpenRouterは `OPENROUTER_API_KEY` / `OPENROUTER_BASE_URL`、Azureは `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT`、Bedrockは `AWS_REGION` / `AWS_DEFAULT_REGION` / `AWS_PROFILE` を参照します。Bedrockは設定ファイル内の `aws_access_key_id` / `aws_secret_access_key` / `aws_session_token` も利用できます。

## Tool configuration

`.maigent/config.yaml` の `tools` でツールを管理できます。

```yaml
tools:
  rag:
    enabled: true
  file_batch:
    enabled: true
    max_output_tokens: 4096
    max_retries: 0
    reasoning_effort: none
  web_search:
    enabled: false
    api_key: tvly-...
    max_results: 5
    timeout_seconds: 10
  sandbox:
    enabled: false
    image: maigent-sandbox:py311
    timeout_seconds: 20
    memory_limit_mb: 512
    pids_limit: 128
    cpus: 1
    install_libraries_on_run: false
    allowed_libraries:
      - beautifulsoup4
      - charset-normalizer
      - lxml
      - matplotlib
      - numpy
      - openpyxl
      - pandas
      - pdfplumber
      - pillow
      - pypdf
      - python-docx
      - python-pptx
      - reportlab
      - requests
      - scipy
      - seaborn
      - tabulate
      - xlsxwriter
      - xlrd
```

`sandbox.enabled` を `true` にすると、計算やPython実行が必要そうなメッセージでDocker sandboxを使います。通常は `install_libraries_on_run: false` のままにして、Dockerイメージ側へ必要なライブラリを事前インストールしてください。`true` にするとsandbox実行ごとに `allowed_libraries` を `pip install` するため、起動が遅くなります。

`memory_limit_mb`(既定512、64〜8192)、`pids_limit`(既定128、16〜2048)、`cpus`(既定1、0.1〜8)は、sandboxコンテナに渡す `docker run --memory` / `--pids-limit` / `--cpus` の値です。暴走コードによるメモリ肥大化やフォーク爆弾からホストを守るための上限で、未設定でも既定値が適用されます。

`web_search.enabled` を `true` にし、`api_key`(または環境変数 `TAVILY_API_KEY`)を設定すると、[Tavily](https://tavily.com/) の検索APIを使って最新情報や外部情報を取得します。APIキー未設定の場合は、計画に選択されても「APIキーが未設定です」という明確なメッセージを返し、黙って失敗しません。`max_results`(既定5、1〜10)は取得件数、`timeout_seconds`(既定10、1〜30)はHTTPタイムアウト秒です。この実装は本リポジトリの開発環境ではTavily APIキーを用意できず、ライブ検索の動作確認までは行えていません。実際のキーを設定した上で一度動作確認することを推奨します。

## Project instructions and skills (AGENTS.md / SKILL.md)

Claude Codeなどで使われる `AGENTS.md` / `SKILL.md` 相当の規約ファイルに対応しています。どちらも `.maigent/` 配下に置くだけで、コード変更や設定なしに反映されます。読み込み順は `.maigent/config.yaml` と同じ3層（`~/.maigent/` → アプリ直下の `.maigent/` → プロジェクト内の `.maigent/`）です。

### `.maigent/AGENTS.md`

プレーンテキストのプロジェクト向け指示書です。存在する層をすべて連結し、`base_instructions.txt` の直後、スレッド要約メモリより前に system instructions へ差し込まれます（[agent/views.py:470](agent/views.py:470) の `_build_instructions()`）。トグルや有効/無効設定はなく、ファイルの有無だけで決まります。

```
.maigent/AGENTS.md
```
```markdown
このプロジェクトでは常に丁寧語で回答してください。
金額は必ず税込表記にしてください。
```

テンプレートは [`.maigent/AGENTS.sample.md`](.maigent/AGENTS.sample.md) を参照してください。`config.yaml.sample` と同じ考え方で、ファイル名が `AGENTS.md` と完全一致しないため実際には読み込まれません。有効化する場合は中身を参考に、同じ `.maigent/` フォルダ内へ `AGENTS.md` という名前で作成してください。

### `.maigent/skills/<name>/SKILL.md`

再利用可能な手順・ガイドラインを、ツール選択（`tool_selector`）に組み込まれた1つの選択肢として登録します。frontmatter（`---`区切り）で `name` / `description` を指定し、本文に実際の指示を書きます。

```
.maigent/skills/weekly-report/SKILL.md
```
```markdown
---
name: weekly-report
description: 週次売上レポートを固定フォーマットで作成する。
---

回答は必ず「Weekly Report:」から始め、箇条書き3点で要点をまとめてください。
```

`tool_selector.enabled` が `true` の場合、このスキルは `skill:weekly-report` という名前でツール一覧に加わり、依頼内容が `description` に合致するとLLMがこれを選択します。選択されると本文（frontmatterより後ろ）がその回答生成の入力に差し込まれ、以後の回答はこの指示に従います。`frontmatter` を省略した場合はディレクトリ名がスキル名になり、`description` は空扱いになります（LLMには「'<name>' スキルの指示に従う」という汎用文言が渡ります）。`SKILL.md` が存在しない、または本文が空のディレクトリは無視されます。同名のスキルがある場合は プロジェクト > アプリ > ユーザー の順で後の層が上書きします。

書き方のテンプレートは [`.maigent/skill/sample/SKILL.md`](.maigent/skill/sample/SKILL.md) を参照してください。ここは意図的に `skills`（複数形）ではなく `skill`（単数形）に置かれており、`load_skills()` の探索対象外なので実際には読み込まれません。実際に使う場合は中身を `.maigent/skills/<スキル名>/SKILL.md` へコピーしてください。

## Sandbox image

Docker sandboxを使う前に、ローカル用イメージを一度ビルドしてください。

```bash
docker build -t maigent-sandbox:py311 docker/sandbox
```

ビルド後、次のコマンドで確認できます。

```bash
docker run --rm maigent-sandbox:py311 python --version
```

`.maigent/config.yaml` の `tools.sandbox.image` が `maigent-sandbox:py311` を指していれば、このアプリはそのイメージを使います。標準の `docker/sandbox/Dockerfile` には、CSV/Excel処理、PDF/Word/PowerPoint処理、グラフ作成、HTML解析、HTTP取得で使う一般的な事務作業向けライブラリを事前インストールしています。ライブラリを追加する場合はDockerfileにも追記し、もう一度 `docker build -t maigent-sandbox:py311 docker/sandbox` を実行してください。実行時インストールが必要な場合だけ `.maigent/config.yaml` で `install_libraries_on_run: true` にします。

Sandboxは許可フォルダを直接マウントしません。ファイル保存が必要な場合、sandbox内のプログラムはstdoutに `maigent_sandbox_result` JSONを出力し、Django側のbrokerが `file_write` feature flag とプロジェクトの書き出し先フォルダを検証してからホスト上に保存します。書き込みはブラウザで指定した書き出し先フォルダ配下に限定され、相対パスはそのフォルダ内として解決されます。これにより、生成コードへ直接書き込み権限を渡さずに成果物だけを保存できます。旧 `maigent_artifacts` 形式は互換用に読み取ります。

RAGで選ばれたCSV/TSV/JSON/TXT/Markdownファイルは、sandboxコード生成時に `rag_1` などのDatasetとして型付きメタ情報を提示します。生成コードはデータ本文を再転記せず、ホスト側が注入する `load_dataset("rag_1")`、`dataset_text("rag_1")`、`dataset_meta("rag_1")` を使います。これにより、CSV区切り文字や列名がLLMの再転記で壊れる問題を避けます。

## Program flow

### 全体構成

```mermaid
flowchart LR
    Browser["Browser UI\nagent/dashboard.html + static/agent/app.js"]
    Views["Django views\nagent/views.py"]
    Models["Django models\nProject / Thread / Message / Settings"]
    Config["RuntimeConfig\nagent/config.py"]
    Planner["Agent planner/tool selector\nagent/tooling.py"]
    Client["LLM client\nagent/openai_client.py"]
    Commands["Slash commands\nagent/slash_commands.py"]
    Access["Path access control\nagent/access.py"]
    Docker["Docker sandbox\nmaigent-sandbox:py311"]
    Files["Allowed local files"]
    LLM["LLM providers\nOpenAI / OpenAI-compatible / Ollama / LM Studio / OpenRouter / Azure / Bedrock"]

    Browser <--> Views
    Views <--> Models
    Views --> Config
    Views --> Commands
    Views --> Planner
    Views --> Client
    Commands --> Access
    Access --> Files
    Planner --> Access
    Planner --> Docker
    Client --> LLM
```

### 起動・初期表示

```mermaid
flowchart TD
    Start["GET / または /threads/<thread_id>/"] --> EnsureDefaults["_ensure_defaults()"]
    EnsureDefaults --> HasProject{"Project exists?"}
    HasProject -- "No" --> CreateDefault["Create Default Project"]
    CreateDefault --> CreateMain["Create Main thread"]
    HasProject -- "Yes" --> PickProject["Use current Project\nor first Project"]
    PickProject --> IsCurrent{"is_current?"}
    IsCurrent -- "No" --> MarkCurrent["Set all Projects is_current=false\nand selected Project=true"]
    IsCurrent -- "Yes" --> PickThread["Pick requested Thread\nor first Project thread"]
    MarkCurrent --> PickThread
    CreateMain --> LoadConfig
    PickThread --> LoadConfig["load_runtime_config(project.path)"]
    LoadConfig --> LoadSettings["Load DB settings\nRAG top_k / final evaluation"]
    LoadSettings --> Render["Render dashboard with\nprojects, threads, messages, tools,\naccess paths, approvals, automations"]
```

### 設定ファイル読み込みとプロバイダ選択

```mermaid
flowchart TD
    Start["load_runtime_config(project_path)"] --> UserConfig["Read ~/.maigent/config.toml|yaml"]
    UserConfig --> AppConfig["Read app .maigent/config.toml|yaml"]
    AppConfig --> ProjectConfig{"project_path configured?"}
    ProjectConfig -- "Yes" --> ReadProject["Read <project>/.maigent/config.toml|yaml"]
    ProjectConfig -- "No" --> EnvModel
    ReadProject --> EnvModel["If model is missing,\napply OPENAI_MODEL"]
    EnvModel --> Runtime["RuntimeConfig(values, sources)"]
    Runtime --> Providers{"providers exists?"}
    Providers -- "No" --> Legacy["Use legacy OpenAI-compatible config\nmodel/api_key/base_url/api_mode"]
    Providers -- "Yes" --> FirstEnabled["Select first enabled provider\nopenai -> openai_compatible -> ollama -> lmstudio -> openrouter -> azure -> bedrock"]
    FirstEnabled --> AnyEnabled{"Any provider enabled?"}
    AnyEnabled -- "No" --> Disabled["No active provider\nmodel resolves empty"]
    AnyEnabled -- "Yes" --> Active["Resolve provider-specific\nmodel, key, endpoint, api_mode,\nheaders, Azure settings, Bedrock settings"]
    Legacy --> Active
    Active --> Redact["Redact sensitive values\nbefore dashboard display"]
```

### LLM client分岐

```mermaid
flowchart TD
    Request["complete_response() / stream_response()"] --> Validate["Validate model and provider settings"]
    Validate --> Provider{"active_provider"}

    Provider -- "openai" --> OpenAICompat["OpenAI-compatible client"]
    Provider -- "openai_compatible" --> OpenAICompat
    Provider -- "ollama" --> OpenAICompat
    Provider -- "lmstudio" --> OpenAICompat
    Provider -- "openrouter" --> OpenAICompat
    Provider -- "azure" --> Azure["AzureOpenAI client"]
    Provider -- "bedrock" --> Bedrock["boto3 bedrock-runtime client"]

    OpenAICompat --> Mode{"api_mode"}
    Mode -- "responses" --> Responses["Responses API"]
    Mode -- "chat" --> Chat["Chat Completions API"]
    Mode -- "auto" --> TryResponses["Try Responses API"]
    TryResponses --> ResponsesOK{"Supported and\nnon-empty output?"}
    ResponsesOK -- "Yes" --> Responses
    ResponsesOK -- "No or empty" --> Chat

    Azure --> AzureChat["Azure Chat Completions\nmodel is deployment name"]
    Bedrock --> BedrockAuth["Use region/profile\nor configured AWS credentials"]
    BedrockAuth --> Converse["Bedrock Converse / ConverseStream"]

    Responses --> Output["Return text deltas and response_id"]
    Chat --> Output
    AzureChat --> Output
    Converse --> Output
```

### メッセージ送信から回答生成

```mermaid
flowchart TD
    Submit["POST /threads/<thread_id>/messages/"] --> ValidateMessage{"message is blank?"}
    ValidateMessage -- "Yes" --> BadRequest["Return 400\nmessage is required"]
    ValidateMessage -- "No" --> SaveUser["Create user Message"]
    SaveUser --> LoadConfig["Load RuntimeConfig"]
    LoadConfig --> Slash{"Starts with / ?"}
    Slash -- "Yes" --> SlashCommand["handle_slash_command()"]
    SlashCommand --> SaveCommandReply["Create assistant Message\nstatus=complete"]
    SaveCommandReply --> CommandJson["Return JSON content"]

    Slash -- "No" --> HasModel{"config.model exists?"}
    HasModel -- "No" --> ModelError["Create assistant Message\nstatus=error"]
    ModelError --> ErrorJson["Return 400"]
    HasModel -- "Yes" --> CreateAssistant["Create assistant Message\nstatus=pending"]
    CreateAssistant --> StreamUrl["Return stream_url"]

    StreamUrl --> SSE["GET stream_url"]
    SSE --> BuildInstructions["Build base instructions\nplus memory summary if enabled"]
    BuildInstructions --> GenerateEval["_generate_with_final_evaluation()"]
    GenerateEval --> SaveAssistant["Save assistant content,\nstatus, provider response_id"]
    SaveAssistant --> Done["Emit SSE done"]
```

### エージェント計画とツール実行

```mermaid
flowchart TD
    GenerateOnce["_generate_once()"] --> Clarifier{"initial_clarifier enabled\nand not an obvious actionable tool plan?"}
    Clarifier -- "Yes" --> ClarifyLLM["Ask LLM for missing-info JSON"]
    ClarifyLLM --> NeedClarify{"needs_clarification\nand questions exist?"}
    NeedClarify -- "Yes" --> ReturnQuestions["Return assistant message\nwith reason and up to 3 questions\nwithout creating AgentRun"]
    NeedClarify -- "No / empty / invalid" --> ToolSelectorCheck
    Clarifier -- "No" --> ToolSelectorCheck{"tool_selector enabled\nand tools available?"}
    ToolSelectorCheck -- "Yes" --> SelectTools["Ask LLM for tool plan JSON\nrag / file_batch / sandbox / web_search / final"]
    SelectTools --> ValidTools{"Valid non-empty plan?"}
    ValidTools -- "Yes" --> LLMPlan["Use LLM-selected plan\nand avoid duplicate final"]
    ValidTools -- "No" --> BuildPlan
    ToolSelectorCheck -- "No" --> BuildPlan["build_agent_plan()"]
    BuildPlan --> WebSearch{"web_search enabled\nand current/external topic?"}
    WebSearch -- "Yes" --> AddWeb["Add web_search step\nTavily API if api_key configured"]
    WebSearch -- "No" --> RagCheck
    AddWeb --> RagCheck{"rag enabled\nand local context likely?"}
    RagCheck -- "Yes" --> AddRag["Add rag step"]
    RagCheck -- "No" --> SandboxCheck
    AddRag --> SandboxCheck{"sandbox enabled\nand computation/code likely?"}
    SandboxCheck -- "Yes" --> AddSandbox["Add sandbox step"]
    SandboxCheck -- "No" --> AnyStep{"Any tool step?"}
    AddSandbox --> AnyStep
    AnyStep -- "No" --> Direct["Add final step"]
    AnyStep -- "Yes" --> MaybeRagLLM
    Direct --> MaybeRagLLM["_apply_llm_rag_decision()\nonly for direct-answer plans with\nallowed local context sources"]
    LLMPlan --> AvoidFailed["_avoid_failed_plan()"]
    MaybeRagLLM --> AvoidFailed
    AvoidFailed --> Execute["_execute_agent_plan()\ncreate AgentState with dynamic queue"]
    Execute --> PopTask["Pop one task from plan_queue"]
    PopTask --> RunTask["Run one tool task"]
    RunTask --> Record["Append TaskExecutionRecord"]
    Record --> Replan["_replan_after_step()\nkeep / replace / finish"]
    Replan -- "keep/replace and queue remains" --> PopTask
    Replan -- "finish or queue empty" --> FinalRoute["_route_final_output()\nsave / discard / add_tasks"]
    FinalRoute -- "add_tasks" --> PopTask
    FinalRoute --> FinalMessage{"Tool returned final message?"}
    FinalMessage -- "Yes" --> ReturnTool["Return tool result as assistant content"]
    FinalMessage -- "No" --> LLMAnswer["Call stream_response()\nfor final answer"]
```

### RAG処理

```mermaid
flowchart TD
    RagStep["RAG step"] --> ShouldSearch{"Explicit path/file request\nor search-like text?"}
    ShouldSearch -- "No" --> NoSearch["Return without search"]
    ShouldSearch -- "Yes" --> HasAccess{"Project has allowed\nread/write paths?"}
    HasAccess -- "No" --> NoContext["Return no_context"]
    HasAccess -- "Yes" --> BuildQuery["Build query from user text\nor LLM RAG decision"]
    BuildQuery --> CollectFiles["Collect allowed files\nwith supported extensions"]
    CollectFiles --> AutoContext{"Direct file/path requested?"}
    AutoContext -- "Yes" --> AttachDirect["Attach direct file context\nup to configured char limit"]
    AutoContext -- "No" --> ListRequest{"File/folder list request?"}
    ListRequest -- "Yes" --> AttachListings["Attach allowed directory listings"]
    ListRequest -- "No" --> BM25["Rank candidates by BM25-like score"]
    BM25 --> Adequate{"Top score >= threshold?"}
    Adequate -- "Yes" --> AttachRanked["Attach top_k ranked snippets"]
    Adequate -- "No" --> Judge{"LLM candidate judge available?"}
    Judge -- "Yes" --> LLMJudge["Ask LLM to select relevant files"]
    Judge -- "No" --> NoContext
    LLMJudge --> Selected{"Relevant files selected?"}
    Selected -- "Yes" --> AttachSelected["Attach selected file context"]
    Selected -- "No" --> NoContext
    AttachDirect --> Context["Return has_context"]
    AttachListings --> Context
    AttachRanked --> Context
    AttachSelected --> Context
```

### Sandbox処理

```mermaid
flowchart TD
    SandboxStep["Sandbox step"] --> CanBuild{"Can build deterministic\nPython locally?"}
    CanBuild -- "Yes" --> LocalProgram["Use extracted Python,\nCSV/stat program, math expression,\nor numeric/count task program"]
    CanBuild -- "No" --> GenerateCode["generate_sandbox_code()\nvia active LLM provider"]
    GenerateCode --> CodeExists{"Generated code exists?"}
    CodeExists -- "No" --> NoProgram["Return Sandbox実行結果: 失敗\nno executable program generated"]
    CodeExists -- "Yes" --> CodeOverride["Pass generated code directly to\nrun_sandbox(..., code=generated_code)"]
    CodeOverride --> LocalProgram
    LocalProgram --> TempScript["Write script.py to temp dir"]
    TempScript --> InstallLibs{"install_libraries_on_run\nand allowed_libraries?"}
    InstallLibs -- "Yes" --> DockerNet["docker run with network\npip install allowed libs\nthen python /work/script.py"]
    InstallLibs -- "No" --> DockerNoNet["docker run --network none\npython /work/script.py"]
    DockerNet --> Result{"Return code 0?"}
    DockerNoNet --> Result
    Result -- "Yes" --> Artifact{"stdout contains typed\nmaigent_sandbox_result?"}
    Artifact -- "Yes" --> Broker["Host broker validates\nfile_write and write access\nthen saves artifacts"]
    Artifact -- "No" --> Success["Return Sandbox実行結果: 成功"]
    Broker --> Success
    Result -- "No" --> Failure{"No executable code?\nor execution failure?"}
    Failure -- "No executable code" --> NoCodeFail["Return Sandbox実行結果: 失敗\nno executable code or expression"]
    Failure -- "Execution failed" --> FailMessage["Return Sandbox実行結果: 失敗\nstdout/stderr"]
```

### 最終評価リトライ

```mermaid
flowchart TD
    Start["_generate_with_final_evaluation()"] --> Settings["Read config values,\nthen DB AppSetting overrides"]
    Settings --> Enabled{"final_evaluation.enabled?"}
    Enabled -- "No" --> OneAttempt["Run one attempt\nand return result"]
    Enabled -- "Yes" --> Attempts["attempts = max_retries + 1"]
    Attempts --> Generate["_generate_once()"]
    Generate --> AssistantError{"assistant status is error?"}
    AssistantError -- "Yes" --> ReturnError["Return error result"]
    AssistantError -- "No" --> Evaluate["_evaluate_final_answer()"]
    Evaluate --> EvalFailed{"Evaluation LLM empty\nor malformed JSON?"}
    EvalFailed -- "Yes" --> KeepCurrent["Keep current answer\nwithout replanning"]
    EvalFailed -- "No" --> Adequate{"adequate?"}
    Adequate -- "Yes" --> ReturnPass["Return answer"]
    Adequate -- "No" --> More{"Attempts left?"}
    More -- "Yes" --> RecordFailure["Record failed plan\nand retry feedback"]
    RecordFailure --> Replan["Replan with different path\nfor example add RAG or remove sandbox"]
    Replan --> Generate
    More -- "No" --> ReturnWarning["Return last answer\nwith final-evaluation warning"]
```

### スラッシュコマンドとファイルアクセス

```mermaid
flowchart TD
    Slash["handle_slash_command()"] --> Command{"Command"}
    Command -- "/status" --> Status["Return project, thread,\nprovider, model, config sources,\nfeature flags"]
    Command -- "/model" --> Model["Return active model"]
    Command -- "/features" --> Features["List/enable/disable FeatureFlag"]
    Command -- "/memories" --> Memories["Toggle Thread.memory_enabled\n(mainly used to disable it)"]
    Command -- "/compact" --> Compact["Summarize latest messages\ninto Thread.summary\nand set memory_enabled=true"]
    Command -- "/resume" --> Resume["List recent project threads"]
    Command -- "/fork" --> Fork["Copy current thread\nand messages to new thread"]
    Command -- "/read" --> Read["Read file"]
    Command -- "/ls" --> List["List folder"]
    Command -- "/write" --> Write["Write file"]
    Command -- "/append" --> Append["Append file"]
    Command -- "/file or /files" --> File{"Path argument exists?"}
    File -- "No" --> AccessList["List allowed access paths"]
    File -- "Yes, directory" --> List
    File -- "Yes, file" --> Read

    Read --> NormalizeRead["Resolve path"]
    List --> NormalizeList["Resolve path"]
    NormalizeRead --> AllowedRead{"is_path_allowed(read)?"}
    NormalizeList --> AllowedList{"is_path_allowed(read)?"}
    AllowedRead -- "No" --> DenyRead["Return read denied"]
    AllowedList -- "No" --> DenyList["Return list denied"]
    AllowedRead -- "Yes" --> ReadText{"Exists, is file,\nUTF-8 text?"}
    ReadText -- "Yes" --> ReturnText["Return text\nmax 12000 chars"]
    ReadText -- "No" --> ReadError["Return specific error"]
    AllowedList -- "Yes" --> IsDir{"Exists and is directory?"}
    IsDir -- "Yes" --> ReturnList["Return up to 80 children"]
    IsDir -- "No" --> ListError["Return specific error"]
    Write --> WriteFlag{"FeatureFlag file_write enabled?"}
    Append --> WriteFlag
    WriteFlag -- "No" --> DenyWriteFlag["Return feature disabled"]
    WriteFlag -- "Yes" --> OutputRoot{"Project output folder configured\nand exists?"}
    OutputRoot -- "No" --> DenyWrite["Return write denied"]
    OutputRoot -- "Yes" --> NormalizeWrite["Resolve relative path under output folder\nor absolute path"]
    NormalizeWrite --> AllowedWrite{"Target is inside output folder?"}
    AllowedWrite -- "No" --> DenyWrite
    AllowedWrite -- "Yes" --> WriteTarget{"Parent exists\nand target is not directory?"}
    WriteTarget -- "No" --> WriteError["Return specific error"]
    WriteTarget -- "Yes" --> WriteFile["Write or append UTF-8 text"]
```

### プロジェクト、スレッド、承認、アクセス許可

```mermaid
flowchart TD
    UI["Dashboard form action"] --> Action{"Action"}
    Action -- "Create Project" --> CreateProject["Create Project\nset as current\ncreate Main thread"]
    Action -- "Switch Project" --> SwitchProject["Set selected Project current\nopen first or new Main thread"]
    Action -- "Create Thread" --> CreateThread["Create Thread under Project"]
    Action -- "Delete Thread" --> DeleteThread["Delete Thread"]
    DeleteThread --> HasThread{"Project still has threads?"}
    HasThread -- "Yes" --> RedirectThread["Redirect to latest thread"]
    HasThread -- "No" --> RecreateMain["Create Main thread\nthen redirect"]
    Action -- "Add Access Path" --> AddPath["Normalize path\nget_or_create ProjectAccessPath"]
    Action -- "Delete Access Path" --> DeletePath["Delete ProjectAccessPath"]
    Action -- "Toggle Feature Flag" --> ToggleFlag["Update FeatureFlag.enabled"]
    Action -- "Create Approval" --> CreateApproval["Create ApprovalRequest\nstatus=pending"]
    Action -- "Approve/Reject" --> ApprovalAction["Update ApprovalRequest.status"]
    Action -- "Save Settings" --> SaveSettings["Clamp rag_top_k and final_evaluation retries\nsave AppSetting"]
```
