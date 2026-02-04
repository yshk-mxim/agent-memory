# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.x.x   | Yes               |
| < 1.0   | No                |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

### How to Report

1. Email the maintainers with a description of the vulnerability
2. Include steps to reproduce the issue
3. If possible, suggest a fix or mitigation

### What to Expect

- **Acknowledgment**: Within 48 hours of your report
- **Assessment**: We will evaluate the severity and impact within 5 business days
- **Resolution**: Critical vulnerabilities will be patched as soon as possible
- **Disclosure**: We will coordinate with you on public disclosure timing

### Scope

The following are in scope for security reports:

- Authentication/authorization bypasses in the API server
- Remote code execution vulnerabilities
- Data exposure through API endpoints
- Dependency vulnerabilities with known exploits
- Path traversal in cache persistence

The following are **out of scope**:

- Denial of service through legitimate API usage (rate limiting is the user's responsibility)
- Issues in dependencies without a known exploit
- Issues requiring physical access to the host machine

## Security Best Practices for Users

- Run the server on a trusted network or behind a reverse proxy
- Do not expose the API server directly to the internet without authentication
- Keep dependencies updated (`pip install --upgrade semantic-cache`)
- Review model configurations before deploying to production
- Use `env.json` for sensitive configuration (API keys) and ensure it is gitignored
