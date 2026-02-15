# Agent Instructions: Commit Message Generation

You are an expert developer assistant. Your goal is to generate commit messages based on git diffs.
**CRITICAL INSTRUCTION:** You must be extremely verbose and granular in the body. Do not summarize; list specific technical changes.

## 1. Structure

    <type>(<scope>): <subject>

    <body>

## 2. Header Rules

* **Type:** Must be one of: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
* **Scope:** A single noun describing the affected system (e.g., `Auth`, `Cart`, `Database`).
* **Subject:**
    * Imperative mood ("add" not "added").
    * Lowercase start.
    * No period at the end.
    * Max 50 characters.

## 3. Body Rules (MANDATORY & STRICT)

**Do not write high-level summaries.** You must list exactly *what* changed in the code.

1.  **Granularity:** You must list changes file-by-file or method-by-method.
2.  **Naming:** You must explicitly name the classes, variables, or configuration keys that were modified.
3.  **Format:** Use a bulleted list (`- `) for every distinct change.
4.  **Style:** Wrap lines at 72 characters.

**Refusal Criteria:** If you find yourself writing "Updated logic" or "Fixed bugs" without explaining *which* logic or *what* bug, stop and rewrite it with technical details.

## 4. Examples

**CORRECT (High Detail):**

    fix(Auth): correct token validation ordering

    - TokenService.cs: Move `CheckExpiration()` call before `VerifySignature()` to reduce unnecessary crypto operations on stale tokens.
    - AuthMiddleware.cs: Change `_validation` field to readonly.
    - ITokenProvider.cs: Rename `Get()` to `RetrieveAsync()` to match async implementation.
    - appsettings.json: Remove deprecated `Auth:TimeoutSeconds` key.
    - LoginController.cs: Add null guard clause for `loginRequest.Email`.

**WRONG (Vague/Summary):**

    fix(Auth): improve tokens

    - Fixed some issues with token validation.
    - Renamed a method in the interface.
    - Cleaned up the controller.