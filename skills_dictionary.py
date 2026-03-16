# Canonical skill names mapped from common aliases/variants.
# Key: lowercase alias → Value: canonical form
# Add new aliases here as you discover them.

SKILL_ALIASES: dict[str, str] = {
    # ── JavaScript / TypeScript ──
    "js": "JavaScript",
    "javascript": "JavaScript",
    "java script": "JavaScript",
    "es6": "JavaScript",
    "ecmascript": "JavaScript",
    "ts": "TypeScript",
    "typescript": "TypeScript",
    "type script": "TypeScript",

    # ── React ──
    "react": "React",
    "reactjs": "React",
    "react.js": "React",
    "react js": "React",
    "reactjss": "React",

    # ── Next.js ──
    "next": "Next.js",
    "nextjs": "Next.js",
    "next.js": "Next.js",
    "next js": "Next.js",

    # ── Node.js ──
    "node": "Node.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "node js": "Node.js",
    "nodej": "Node.js",

    # ── Express ──
    "express": "Express.js",
    "expressjs": "Express.js",
    "express.js": "Express.js",

    # ── Python ──
    "python": "Python",
    "python3": "Python",
    "py": "Python",
    "py3": "Python",

    # ── Django ──
    "django": "Django",
    "djano": "Django",
    "dajngo": "Django",

    # ── Flask ──
    "flask": "Flask",

    # ── FastAPI ──
    "fastapi": "FastAPI",
    "fast api": "FastAPI",
    "fast-api": "FastAPI",

    # ── Java ──
    "java": "Java",

    # ── Spring Boot ──
    "spring": "Spring",
    "spring boot": "Spring Boot",
    "springboot": "Spring Boot",
    "spring-boot": "Spring Boot",

    # ── C / C++ / C# ──
    "c": "C",
    "c++": "C++",
    "cpp": "C++",
    "cplusplus": "C++",
    "c#": "C#",
    "csharp": "C#",
    "c sharp": "C#",

    # ── Go ──
    "go": "Go",
    "golang": "Go",

    # ── Rust ──
    "rust": "Rust",
    "rustlang": "Rust",

    # ── Ruby ──
    "ruby": "Ruby",
    "ruby on rails": "Ruby on Rails",
    "rails": "Ruby on Rails",
    "ror": "Ruby on Rails",

    # ── PHP ──
    "php": "PHP",
    "laravel": "Laravel",

    # ── Swift / Kotlin ──
    "swift": "Swift",
    "kotlin": "Kotlin",
    "kt": "Kotlin",

    # ── Angular ──
    "angular": "Angular",
    "angularjs": "Angular",
    "angular.js": "Angular",
    "angular js": "Angular",

    # ── Vue ──
    "vue": "Vue.js",
    "vuejs": "Vue.js",
    "vue.js": "Vue.js",
    "vue js": "Vue.js",
    "vue3": "Vue.js",
    "vue 3": "Vue.js",

    # ── Svelte ──
    "svelte": "Svelte",
    "sveltejs": "Svelte",
    "sveltekit": "SvelteKit",

    # ── CSS / Styling ──
    "css": "CSS",
    "css3": "CSS",
    "html": "HTML",
    "html5": "HTML",
    "sass": "Sass",
    "scss": "Sass",
    "less": "Less",
    "tailwind": "Tailwind CSS",
    "tailwindcss": "Tailwind CSS",
    "tailwind css": "Tailwind CSS",
    "bootstrap": "Bootstrap",
    "material ui": "Material UI",
    "materialui": "Material UI",
    "mui": "Material UI",
    "chakra ui": "Chakra UI",
    "chakraui": "Chakra UI",
    "styled components": "Styled Components",
    "styled-components": "Styled Components",

    # ── State Management ──
    "redux": "Redux",
    "reduxjs": "Redux",
    "redux toolkit": "Redux Toolkit",
    "rtk": "Redux Toolkit",
    "zustand": "Zustand",
    "mobx": "MobX",
    "recoil": "Recoil",
    "jotai": "Jotai",

    # ── Databases ──
    "sql": "SQL",
    "mysql": "MySQL",
    "my sql": "MySQL",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "psql": "PostgreSQL",
    "mongo": "MongoDB",
    "mongodb": "MongoDB",
    "mongo db": "MongoDB",
    "redis": "Redis",
    "sqlite": "SQLite",
    "dynamodb": "DynamoDB",
    "dynamo db": "DynamoDB",
    "cassandra": "Cassandra",
    "mariadb": "MariaDB",
    "couchdb": "CouchDB",
    "firestore": "Firestore",
    "firebase": "Firebase",
    "supabase": "Supabase",

    # ── ORM / Query Builders ──
    "prisma": "Prisma",
    "sequelize": "Sequelize",
    "typeorm": "TypeORM",
    "drizzle": "Drizzle ORM",
    "mongoose": "Mongoose",
    "sqlalchemy": "SQLAlchemy",
    "sql alchemy": "SQLAlchemy",

    # ── Cloud / DevOps ──
    "aws": "AWS",
    "amazon web services": "AWS",
    "gcp": "Google Cloud Platform",
    "google cloud": "Google Cloud Platform",
    "azure": "Microsoft Azure",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "terraform": "Terraform",
    "ansible": "Ansible",
    "jenkins": "Jenkins",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "ci cd": "CI/CD",
    "github actions": "GitHub Actions",
    "gitlab ci": "GitLab CI",
    "nginx": "Nginx",
    "apache": "Apache",
    "heroku": "Heroku",
    "vercel": "Vercel",
    "netlify": "Netlify",
    "digitalocean": "DigitalOcean",

    # ── Version Control ──
    "git": "Git",
    "github": "GitHub",
    "gitlab": "GitLab",
    "bitbucket": "Bitbucket",
    "svn": "SVN",

    # ── Testing ──
    "jest": "Jest",
    "mocha": "Mocha",
    "cypress": "Cypress",
    "playwright": "Playwright",
    "selenium": "Selenium",
    "pytest": "pytest",
    "unittest": "unittest",
    "vitest": "Vitest",
    "rtl": "React Testing Library",
    "react testing library": "React Testing Library",

    # ── API / Communication ──
    "rest": "REST",
    "rest api": "REST",
    "restful": "REST",
    "graphql": "GraphQL",
    "graph ql": "GraphQL",
    "grpc": "gRPC",
    "websocket": "WebSocket",
    "websockets": "WebSocket",
    "socket.io": "Socket.IO",
    "socketio": "Socket.IO",

    # ── Build / Bundlers ──
    "webpack": "Webpack",
    "vite": "Vite",
    "rollup": "Rollup",
    "esbuild": "esbuild",
    "parcel": "Parcel",
    "babel": "Babel",
    "swc": "SWC",
    "turbopack": "Turbopack",

    # ── Package Managers ──
    "npm": "npm",
    "yarn": "Yarn",
    "pnpm": "pnpm",
    "pip": "pip",
    "uv": "uv",
    "poetry": "Poetry",

    # ── Data / ML ──
    "machine learning": "Machine Learning",
    "ml": "Machine Learning",
    "deep learning": "Deep Learning",
    "dl": "Deep Learning",
    "ai": "Artificial Intelligence",
    "artificial intelligence": "Artificial Intelligence",
    "tensorflow": "TensorFlow",
    "tf": "TensorFlow",
    "pytorch": "PyTorch",
    "torch": "PyTorch",
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "numpy": "NumPy",
    "scipy": "SciPy",
    "matplotlib": "Matplotlib",
    "nlp": "NLP",
    "natural language processing": "NLP",
    "opencv": "OpenCV",
    "computer vision": "Computer Vision",
    "langchain": "LangChain",
    "llm": "LLM",
    "large language model": "LLM",
    "openai": "OpenAI API",
    "huggingface": "Hugging Face",
    "hugging face": "Hugging Face",

    # ── Mobile ──
    "react native": "React Native",
    "reactnative": "React Native",
    "rn": "React Native",
    "flutter": "Flutter",
    "dart": "Dart",
    "swiftui": "SwiftUI",
    "jetpack compose": "Jetpack Compose",
    "android": "Android",
    "ios": "iOS",

    # ── Misc Tools ──
    "linux": "Linux",
    "bash": "Bash",
    "shell": "Shell Scripting",
    "powershell": "PowerShell",
    "figma": "Figma",
    "jira": "Jira",
    "confluence": "Confluence",
    "slack": "Slack",
    "postman": "Postman",
    "swagger": "Swagger",
    "elasticsearch": "Elasticsearch",
    "elastic search": "Elasticsearch",
    "kafka": "Apache Kafka",
    "rabbitmq": "RabbitMQ",
    "rabbit mq": "RabbitMQ",
    "graphite": "Graphite",
    "grafana": "Grafana",
    "prometheus": "Prometheus",
    "datadog": "Datadog",
    "sentry": "Sentry",

    # ── Soft Skills (common in resumes) ──
    "leadership": "Leadership",
    "communication": "Communication",
    "teamwork": "Teamwork",
    "problem solving": "Problem Solving",
    "problem-solving": "Problem Solving",
    "agile": "Agile",
    "scrum": "Scrum",
    "kanban": "Kanban",
    "project management": "Project Management",
    "time management": "Time Management",
}

# Reverse map: canonical → set of aliases (auto-built, used for fuzzy target list)
CANONICAL_SKILLS: set[str] = set(SKILL_ALIASES.values())
