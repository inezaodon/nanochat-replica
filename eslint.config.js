import js from "@eslint/js";
import tseslint from "typescript-eslint";

export default [
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    ignores: ["dist/**", "node_modules/**", ".venv/**"],
  },
  {
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
    },
  },
  {
    files: ["scripts/**/*.mjs", "scripts/**/*.cjs", "scripts/**/*.js"],
    languageOptions: {
      globals: {
        Buffer: "readonly",
        __dirname: "readonly",
        __filename: "readonly",
        console: "readonly",
        fetch: "readonly",
        process: "readonly",
        setImmediate: "readonly",
        clearImmediate: "readonly",
      },
    },
  },
];

