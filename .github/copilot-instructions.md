# Git Commit Message Instructions

When generating git commit messages, you must follow these rules:

1. Format: Use the Conventional Commits specification: `type(scope): description`.
2. Types allowed:
   - `feat`: A new feature
   - `fix`: A bug fix
   - `docs`: Documentation changes
   - `style`: Code style changes (whitespace, formatting, missing semi-colons)
   - `refactor`: Code changes that neither fix a bug nor add a feature
   - `chore`: Updating build tasks, package manager configs, etc.
3. Length: Limit the first line (Summary) to 50 characters or less.
4. Letter Case: Use lowercase for the type, scope, and the start of the description.
5. Mood: Use the imperative present tense (e.g., "add" instead of "added", "fix" instead of "fixes").
6. Body: List structural details in a concise, bulleted list only if the changes are complex. 
