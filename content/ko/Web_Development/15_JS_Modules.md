# JavaScript 모듈 시스템

## 학습 목표
- ES Modules (ESM)의 import/export 문법 이해
- CommonJS와 ESM의 차이점 파악
- 동적 import와 코드 스플리팅 활용
- 모듈 번들러의 역할 이해

---

## 1. 모듈의 필요성

### 1.1 모듈이란?

```
┌─────────────────────────────────────────────────────────────────┐
│                    모듈의 장점                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Before (전역 스코프 오염):                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ <script src="lib1.js"></script>  <!-- var helper = ... --> │  │
│  │ <script src="lib2.js"></script>  <!-- var helper = ... --> │  │
│  │ <script src="app.js"></script>   <!-- helper? 어떤 것? -->  │  │
│  │                                                            │  │
│  │ 문제: 이름 충돌, 의존성 순서, 전역 변수 오염               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  After (모듈 시스템):                                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ // lib1.js                                                 │  │
│  │ export const helper = () => { ... };                       │  │
│  │                                                            │  │
│  │ // app.js                                                  │  │
│  │ import { helper } from './lib1.js';                        │  │
│  │                                                            │  │
│  │ 장점: 명확한 의존성, 캡슐화, 재사용성                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 모듈 시스템 종류

```javascript
// 1. CommonJS (Node.js 기본)
const fs = require('fs');
module.exports = { myFunction };

// 2. ES Modules (ECMAScript 표준, 브라우저 + Node.js)
import fs from 'fs';
export const myFunction = () => {};

// 3. AMD (RequireJS, 레거시)
define(['dependency'], function(dep) {
    return { myFunction };
});

// 4. UMD (Universal, 호환성 목적)
(function(root, factory) {
    if (typeof define === 'function' && define.amd) {
        define(['dep'], factory);
    } else if (typeof module === 'object') {
        module.exports = factory(require('dep'));
    } else {
        root.myModule = factory(root.dep);
    }
}(this, function(dep) { ... }));
```

---

## 2. ES Modules (ESM) 기본

### 2.1 Export 문법

```javascript
// ==============================
// math.js - Named Exports
// ==============================

// 개별 내보내기
export const PI = 3.14159;

export function add(a, b) {
    return a + b;
}

export function subtract(a, b) {
    return a - b;
}

export class Calculator {
    add(a, b) { return a + b; }
    subtract(a, b) { return a - b; }
}

// 일괄 내보내기
const multiply = (a, b) => a * b;
const divide = (a, b) => a / b;

export { multiply, divide };

// 이름 변경 내보내기
const internalName = () => 'internal';
export { internalName as publicName };


// ==============================
// utils.js - Default Export
// ==============================

// 기본 내보내기 (파일당 1개)
export default function formatDate(date) {
    return date.toISOString().split('T')[0];
}

// 또는
function formatDate(date) {
    return date.toISOString().split('T')[0];
}
export default formatDate;

// 또는 (익명 함수)
export default function(date) {
    return date.toISOString().split('T')[0];
}


// ==============================
// 혼합 사용 (권장)
// ==============================

// api.js
export const API_URL = 'https://api.example.com';
export const fetchData = async (endpoint) => { /* ... */ };

// 기본 내보내기도 함께
export default class ApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }
}
```

### 2.2 Import 문법

```javascript
// ==============================
// Named Imports
// ==============================

// 개별 가져오기
import { add, subtract } from './math.js';
console.log(add(1, 2));  // 3

// 이름 변경 가져오기
import { add as sum } from './math.js';
console.log(sum(1, 2));  // 3

// 전체 가져오기 (네임스페이스)
import * as math from './math.js';
console.log(math.add(1, 2));  // 3
console.log(math.PI);         // 3.14159


// ==============================
// Default Imports
// ==============================

// 기본 가져오기 (이름 자유)
import formatDate from './utils.js';
import myFormatter from './utils.js';  // 같은 것, 다른 이름

console.log(formatDate(new Date()));


// ==============================
// 혼합 Import
// ==============================

// Default + Named 함께
import ApiClient, { API_URL, fetchData } from './api.js';

// 또는
import ApiClient from './api.js';
import { API_URL, fetchData } from './api.js';


// ==============================
// 부수 효과 Import (Side Effects)
// ==============================

// 코드 실행만 (polyfill, 스타일 등)
import './polyfill.js';
import './styles.css';  // 번들러 사용 시


// ==============================
// Re-export
// ==============================

// index.js (배럴 파일)
export { add, subtract } from './math.js';
export { default as formatDate } from './utils.js';
export * from './helpers.js';  // 모든 named exports
```

### 2.3 HTML에서 사용

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>ES Modules</title>
</head>
<body>
    <!-- type="module" 필수 -->
    <script type="module">
        import { add } from './math.js';
        console.log(add(1, 2));
    </script>

    <!-- 외부 모듈 파일 -->
    <script type="module" src="./app.js"></script>

    <!-- 비모듈 폴백 (구형 브라우저) -->
    <script nomodule src="./app-legacy.js"></script>
</body>
</html>
```

---

## 3. 동적 Import

### 3.1 기본 문법

```javascript
// 정적 import (파일 상단, 파싱 시점)
import { add } from './math.js';

// 동적 import (런타임, Promise 반환)
async function loadMath() {
    const math = await import('./math.js');
    console.log(math.add(1, 2));  // 3
}

// 또는 then 사용
import('./math.js').then(math => {
    console.log(math.add(1, 2));
});

// default export 접근
const module = await import('./utils.js');
const formatDate = module.default;
```

### 3.2 조건부 로딩

```javascript
// 사용자 권한에 따른 모듈 로딩
async function loadAdminPanel() {
    if (user.isAdmin) {
        const { AdminPanel } = await import('./admin.js');
        return new AdminPanel();
    }
    return null;
}

// 기능 감지 후 로딩
async function loadPolyfill() {
    if (!window.IntersectionObserver) {
        await import('intersection-observer');
    }
}

// 라우트 기반 로딩
const routes = {
    '/': () => import('./pages/Home.js'),
    '/about': () => import('./pages/About.js'),
    '/contact': () => import('./pages/Contact.js'),
};

async function loadPage(path) {
    const loader = routes[path];
    if (loader) {
        const module = await loader();
        return module.default;
    }
}
```

### 3.3 코드 스플리팅

```javascript
// React에서 lazy loading
import React, { lazy, Suspense } from 'react';

const HeavyComponent = lazy(() => import('./HeavyComponent'));

function App() {
    return (
        <Suspense fallback={<div>Loading...</div>}>
            <HeavyComponent />
        </Suspense>
    );
}

// Vue에서 비동기 컴포넌트
const AsyncComponent = () => ({
    component: import('./AsyncComponent.vue'),
    loading: LoadingComponent,
    error: ErrorComponent,
    delay: 200,
    timeout: 3000
});

// 순수 JavaScript
class Router {
    async loadRoute(path) {
        const pageModule = await import(`./pages/${path}.js`);
        const Page = pageModule.default;
        this.render(new Page());
    }
}
```

---

## 4. CommonJS vs ES Modules

### 4.1 주요 차이점

```
┌──────────────────────────────────────────────────────────────────┐
│                CommonJS vs ES Modules 비교                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  특성          CommonJS              ES Modules                  │
│  ────────────────────────────────────────────────────────────── │
│  문법          require/exports        import/export               │
│  로딩 시점      런타임 (동기)          파싱 시점 (정적 분석)       │
│  기본 환경      Node.js               브라우저 + Node.js          │
│  동적 로딩      require() 어디서든    import() 별도 문법          │
│  Tree Shaking   어려움                 가능                       │
│  순환 참조      부분 지원              더 나은 지원                │
│  파일 확장자    .js (기본)             .mjs 또는 type="module"     │
│  this 값        module.exports        undefined                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 코드 비교

```javascript
// ==============================
// CommonJS
// ==============================

// utils.cjs
const helper = () => 'helper';
const PI = 3.14159;

module.exports = { helper, PI };
// 또는
exports.helper = helper;
exports.PI = PI;

// app.cjs
const { helper, PI } = require('./utils.cjs');
const utils = require('./utils.cjs');
console.log(utils.PI);


// ==============================
// ES Modules
// ==============================

// utils.mjs
export const helper = () => 'helper';
export const PI = 3.14159;

// app.mjs
import { helper, PI } from './utils.mjs';
import * as utils from './utils.mjs';
console.log(utils.PI);
```

### 4.3 Node.js에서 ESM 사용

```json
// package.json
{
    "name": "my-package",
    "type": "module",  // 전체 프로젝트를 ESM으로
    "exports": {
        ".": {
            "import": "./dist/index.mjs",
            "require": "./dist/index.cjs"
        }
    }
}
```

```javascript
// .mjs 확장자 사용 (type 무관)
// utils.mjs
export const hello = () => 'Hello';

// app.mjs
import { hello } from './utils.mjs';

// CommonJS에서 ESM import
// (Node.js 14+, top-level await 없이)
async function main() {
    const { hello } = await import('./utils.mjs');
    console.log(hello());
}
main();
```

---

## 5. 모듈 패턴

### 5.1 배럴 파일 (Barrel)

```javascript
// components/index.js (배럴 파일)
export { default as Button } from './Button.js';
export { default as Input } from './Input.js';
export { default as Modal } from './Modal.js';
export { Card, CardHeader, CardBody } from './Card.js';

// 사용 측
import { Button, Input, Modal, Card } from './components';
// 대신
// import Button from './components/Button.js';
// import Input from './components/Input.js';
// ...
```

### 5.2 팩토리 패턴

```javascript
// logger.js
export function createLogger(prefix) {
    return {
        log: (msg) => console.log(`[${prefix}] ${msg}`),
        error: (msg) => console.error(`[${prefix}] ERROR: ${msg}`),
        warn: (msg) => console.warn(`[${prefix}] WARNING: ${msg}`),
    };
}

// 사용
import { createLogger } from './logger.js';
const logger = createLogger('App');
logger.log('Started');  // [App] Started
```

### 5.3 싱글톤 패턴

```javascript
// config.js (모듈 자체가 싱글톤)
let instance = null;

class Config {
    constructor() {
        if (instance) {
            return instance;
        }
        this.settings = {};
        instance = this;
    }

    set(key, value) {
        this.settings[key] = value;
    }

    get(key) {
        return this.settings[key];
    }
}

export default new Config();

// 사용 (항상 같은 인스턴스)
import config from './config.js';
config.set('api_url', 'https://api.example.com');
```

### 5.4 플러그인 패턴

```javascript
// core.js
class App {
    constructor() {
        this.plugins = [];
    }

    use(plugin) {
        plugin.install(this);
        this.plugins.push(plugin);
        return this;  // 체이닝
    }
}

export default new App();

// plugins/logger.js
export default {
    install(app) {
        app.log = (msg) => console.log(`[App] ${msg}`);
    }
};

// plugins/analytics.js
export default {
    install(app) {
        app.track = (event) => console.log(`Track: ${event}`);
    }
};

// main.js
import app from './core.js';
import logger from './plugins/logger.js';
import analytics from './plugins/analytics.js';

app.use(logger).use(analytics);
app.log('Hello');     // [App] Hello
app.track('pageview'); // Track: pageview
```

---

## 6. 번들러와 모듈

### 6.1 번들러의 역할

```
┌─────────────────────────────────────────────────────────────────┐
│                    번들러 (Bundler)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  입력:                        출력:                              │
│  ┌─────────┐                 ┌─────────────────────────────┐   │
│  │ app.js  │                 │                             │   │
│  │ └─ a.js │   ──────────▶  │ bundle.js (하나의 파일)     │   │
│  │ └─ b.js │    번들러       │                             │   │
│  │ └─ c.js │                 │ + chunk1.js (코드 스플리팅) │   │
│  └─────────┘                 │ + chunk2.js                 │   │
│                              └─────────────────────────────┘   │
│                                                                 │
│  주요 기능:                                                      │
│  - 의존성 분석 및 해결                                           │
│  - 코드 변환 (Babel, TypeScript)                                │
│  - 코드 스플리팅                                                 │
│  - Tree Shaking (사용하지 않는 코드 제거)                         │
│  - 최소화 (Minification)                                         │
│  - 에셋 처리 (CSS, 이미지)                                       │
│                                                                 │
│  대표 번들러: Vite, webpack, esbuild, Rollup, Parcel            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Vite 예제

```javascript
// vite.config.js
import { defineConfig } from 'vite';

export default defineConfig({
    build: {
        rollupOptions: {
            output: {
                // 수동 청크 설정
                manualChunks: {
                    vendor: ['react', 'react-dom'],
                    utils: ['lodash', 'date-fns'],
                },
            },
        },
    },
});

// 동적 import는 자동으로 청크 분리
const AdminPage = () => import('./pages/Admin.js');
```

### 6.3 Tree Shaking

```javascript
// utils.js
export function usedFunction() {
    return 'used';
}

export function unusedFunction() {  // 번들에서 제거됨
    return 'unused';
}

// app.js
import { usedFunction } from './utils.js';
// unusedFunction은 import하지 않음

console.log(usedFunction());

// 번들 결과 (Tree Shaking 적용)
// unusedFunction 코드는 포함되지 않음
```

---

## 7. 실전 프로젝트 구조

### 7.1 추천 구조

```
project/
├── src/
│   ├── index.js          # 진입점
│   ├── app.js            # 메인 앱 로직
│   │
│   ├── components/       # UI 컴포넌트
│   │   ├── index.js      # 배럴 파일
│   │   ├── Button.js
│   │   ├── Modal.js
│   │   └── Card/
│   │       ├── index.js
│   │       ├── Card.js
│   │       └── Card.css
│   │
│   ├── utils/            # 유틸리티 함수
│   │   ├── index.js
│   │   ├── format.js
│   │   └── validation.js
│   │
│   ├── services/         # API 호출
│   │   ├── index.js
│   │   ├── api.js
│   │   └── auth.js
│   │
│   ├── store/            # 상태 관리
│   │   ├── index.js
│   │   └── userStore.js
│   │
│   └── constants/        # 상수
│       └── index.js
│
├── public/
│   └── index.html
│
├── package.json
└── vite.config.js
```

### 7.2 Import 경로 별칭

```javascript
// vite.config.js
import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src'),
            '@components': path.resolve(__dirname, './src/components'),
            '@utils': path.resolve(__dirname, './src/utils'),
        },
    },
});

// 사용
import { Button } from '@components';
import { formatDate } from '@utils';
// 대신
// import { Button } from '../../../components';
```

---

## 정리

### ESM 핵심 문법

| 기능 | 문법 |
|------|------|
| Named Export | `export const foo = ...` |
| Default Export | `export default ...` |
| Named Import | `import { foo } from '...'` |
| Default Import | `import foo from '...'` |
| 전체 Import | `import * as mod from '...'` |
| 동적 Import | `await import('...')` |
| Re-export | `export { foo } from '...'` |

### 모범 사례

1. **Named Export 선호**: IDE 자동완성, Tree Shaking 용이
2. **배럴 파일 사용**: 깔끔한 import 경로
3. **순환 참조 피하기**: 의존성 방향 정리
4. **동적 import 활용**: 큰 모듈 지연 로딩

### 다음 단계
- [13_Build_Tools_Environment.md](./13_Build_Tools_Environment.md): 빌드 도구 (Vite, webpack)

---

## 참고 자료

- [MDN JavaScript Modules](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules)
- [Node.js ESM Documentation](https://nodejs.org/api/esm.html)
- [Vite Guide](https://vitejs.dev/guide/)
- [JavaScript Module Pattern](https://www.patterns.dev/posts/module-pattern)
