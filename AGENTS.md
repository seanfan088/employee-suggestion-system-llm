# AGENTS GUIDE FOR THIS REPOSITORY

---

## 1. Build, Lint, and Test Commands

### Build
- No explicit build scripts detected in the package.json or workspace.
- Generally, this repo seems to be TypeScript-based (due to `.ts` files and usage of Zod).
- Use your local environment tooling to compile if necessary (e.g., `tsc`), however no explicit build step is required or defined.

### Lint
- No linter configuration or scripts found (
`eslint`, `prettier`, etc.)
- Follow default TypeScript linting and style guidelines consistent with code style section below.

### Testing
- Uses the `vitest` test framework.
- To run all tests: 
  ```
  npx vitest
  ```
- To run a single test file:
  ```
  npx vitest path/to/single_test_file.ts
  ```
- To run tests matching a pattern:
  ```
  npx vitest --testNamePattern="pattern"
  ```
- For more options and watch mode:
  ```
  npx vitest --watch
  ```

---

## 2. Code Style Guidelines

### Language and Types
- Code is primarily written in TypeScript.
- Use strong typing consistently.
- Use Zod for schema validation and type inference.
- Use `z.infer<typeof schema>` for type inference.
- Prefer immutable data structures where possible.

### Imports
- Use ES module import syntax.
- Separate external imports from internal imports with a blank line.
- Import types explicitly when needed.
- Group similar imports together for readability.

### Formatting
- Prefer using 2 spaces per indentation level.
- Use semicolons at end of statements.
- Use single quotes for strings unless template literals or double quotes needed.
- Use trailing commas in multi-line objects and arrays.
- Keep lines under 120 characters.
- Use consistent line breaks between logical code blocks.

### Naming Conventions
- Use camelCase for variable and function names.
- Use PascalCase for types, interfaces, classes, and Zod schemas.
- Constants and enums can be SCREAMING_SNAKE_CASE or PascalCase.

### Functions
- Use descriptive function and variable names.
- Keep functions short and focused.

### Error Handling
- Use Zod's built-in error mechanisms for validation.
- Throw errors when validation fails using Zod's parse or safeParse functions.
- Catch errors in async functions as needed.
- Provide useful error messages in Zod schema definitions.

### Comments and Documentation
- Use JSDoc style comments for public functions and complex logic.
- Document assumptions and constraints.

### Tests
- Place tests adjacent to logic or in parallel test folders following Vitest convention.
- Use descriptive test names.
- Utilize `expect` and `test` from `vitest`.
- Write both synchronous and asynchronous tests where appropriate.

---

## 3. Additional Notes

### Cursor Rules
- No `.cursor/rules/` or `.cursorrules` found in the project.

### Copilot Rules
- No `.github/copilot-instructions.md` or similar detected.

### Dependency
- Uses `@opencode-ai/plugin` as dependency (should be respected).
- Heavy use of Zod library for schema validation.

---

## 4. Summary of Key Commands

```bash
# Run all tests
npx vitest

# Run a specific test file
npx vitest src/path/to/testfile.test.ts

# Run tests matching a name pattern
npx vitest --testNamePattern="pattern"

# Run tests in watch mode
npx vitest --watch
```

---

## 5. Recommendations for Agents

- Use TypeScript features and Zod schemas for safety and validation.
- Follow the naming and formatting conventions strictly for consistency.
- Ensure error messages are meaningful in validations.
- Run tests frequently during changes with Vitest.
- No automatic lint or build step is mandated; follow established TypeScript best practices.

---

*This file was generated based on codebase analysis and available project metadata.*
