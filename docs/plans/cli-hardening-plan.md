# CLI Hardening Implementation Plan
> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
**Goal:** Transform CLI from basic to production-ready with comprehensive testing, error handling, and user experience improvements.
**Architecture:** Add testing infrastructure, error types, progress tracking, input validation, signal handling, and configuration management.
**Tech Stack:** Python/Click, pytest, Rich, tqdm, signal handling, typer (optional upgrade)

## Phase 1: Foundation (High Priority)

### CLI-1: Add CLI Testing Infrastructure
**Issue:** No CLI testing means commands break silently
**Implementation:**
- Create `tests/test_cli/` directory structure
- Add CLI-specific test modules:
  - `test_cli_commands.py` - Command functionality tests
  - `test_cli_integration.py` - End-to-end integration tests
  - `test_cli_error_handling.py` - Error scenario tests
- Mock backend dependencies for fast tests
- Add CLI test utilities and fixtures
- Configure pytest to run CLI tests
**Expected Result:** CLI commands have test coverage to prevent regressions

### CLI-2: Implement Better Error Handling
**Issue:** All errors handled the same way with generic messages
**Implementation:**
- Create custom exception hierarchy:
  - `CoarchCLIError` (base)
  - `CoarchConfigError` (configuration issues)
  - `CoarchIndexError` (indexing problems)
  - `CoarchSearchError` (search failures)
  - `CoarchValidationError` (input validation)
- Add user-friendly error messages with recovery suggestions
- Implement different exit codes for different error types
- Add error context and debugging information
- Replace generic Exception catches with specific handling
**Expected Result:** Users get actionable error messages instead of cryptic exceptions

### CLI-3: Add Progress Bars for Long Operations
**Issue:** Users can't track progress of long operations (embedding 1000s of chunks)
**Implementation:**
- Add tqdm-based progress bars for:
  - File indexing progress
  - Embedding generation batches
  - FAISS index building
  - Repository deletion operations
- Add ETA estimates for long operations
- Implement conditional progress bars (show only if operation > 5 seconds)
- Add progress callbacks for chunked operations
- Format progress output consistently
**Expected Result:** Users see real-time progress and estimated completion times

### CLI-4: Add Input Validation and Sanitization
**Issue:** No validation of user inputs leads to runtime failures
**Implementation:**
- Add comprehensive input validation:
  - Path validation (exists, readable, writable)
  - Limit validation (positive integers, reasonable bounds)
  - Query validation (length, special characters)
  - Port validation (1-65535, available)
- Add path sanitization (traversal protection)
- Add configuration validation on startup
- Add early failure detection before expensive operations
- Implement validation decorators for Click options
**Expected Result:** Fail fast with clear messages instead of cryptic errors

## Phase 2: Reliability (Medium Priority)

### CLI-5: Implement Signal Handling for Graceful Shutdown
**Issue:** No graceful shutdown, resource leaks on interrupts
**Implementation:**
- Add signal handlers for SIGINT, SIGTERM
- Implement graceful shutdown with cleanup:
  - Save partial progress
  - Close file handles and connections
  - Stop background threads
  - Clean up temporary files
- Add shutdown timeout (kill after 10 seconds if not clean)
- Implement cancellation support for long operations
- Add "are you sure" prompts for destructive operations
**Expected Result:** Clean shutdowns preserve data and resources

### CLI-6: Add Command Aliases and Tab Completion
**Issue:** Limited discoverability and productivity features
**Implementation:**
- Add command aliases:
  - `search` = `find`
  - `status` = `stats`
  - `serve` = `server`
- Implement shell completion for:
  - Command names
  - File paths
  - Language options
  - Configuration values
- Add context-aware completion (e.g., show only directories for `index`)
- Support bash, zsh, and fish completion
**Expected Result:** Faster command entry and better discoverability

### CLI-7: Improve Configuration Management
**Issue:** Configuration scattered across multiple sources
**Implementation:**
- Consolidate configuration handling:
  - CLI options (highest priority)
  - Environment variables
  - Configuration files
  - Default values (lowest priority)
- Add config validation and conflict detection
- Implement config templates for quick setup
- Add `--print-config` to show resolved configuration
- Add config migration for version upgrades
- Implement config history and rollback
**Expected Result:** Predictable configuration behavior and easy setup

## Phase 3: Polish (Low Priority)

### CLI-8: Add CLI Integration Tests
**Issue:** End-to-end workflow testing is manual
**Implementation:**
- Create integration test scenarios:
  - Full indexing workflow
  - Search functionality
  - Server lifecycle
  - Error recovery scenarios
- Add test repository fixtures
- Implement automated smoke tests
- Add performance regression tests
- Add cross-platform compatibility tests
- Generate test reports and coverage metrics
**Expected Result:** Confident deployments and regression detection

## Expected Outcomes

### Before (Current State)
- ✅ Basic functionality works
- ❌ No error handling
- ❌ No progress tracking
- ❌ No input validation
- ❌ No testing
- ❌ No graceful shutdown

### After (Target State)
- ✅ Comprehensive error handling with actionable messages
- ✅ Real-time progress tracking with ETA estimates
- ✅ Input validation with early failure detection
- ✅ Full CLI test coverage preventing regressions
- ✅ Graceful shutdown preserving data and resources
- ✅ Command aliases and tab completion for productivity
- ✅ Unified configuration management

### Metrics
- **Robustness Score:** 6/10 → 9/10
- **Test Coverage:** 0% → 85%
- **Error Recovery:** Manual → Automatic
- **User Experience:** Functional → Professional
- **Production Readiness:** Fragile → Enterprise-grade

## Implementation Notes
1. **Backwards Compatibility:** Maintain all existing command interfaces
2. **Performance:** No regression in CLI startup time or command execution
3. **Dependencies:** Use existing packages (tqdm, pytest) where possible
4. **Testing:** Add tests incrementally with each feature
5. **Documentation:** Update CLI help and examples with new features

## Risk Assessment
- **Low Risk:** All changes are additive, no breaking changes
- **Medium Risk:** Signal handling requires careful coordination with backend
- **High Reward:** Significant improvement in production reliability and UX