#!/usr/bin/env python3
"""Demo script for CLI-6: Command Aliases and Tab Completion."""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli.main import (
    SUPPORTED_LANGUAGES,
    COMMAND_ALIASES,
    get_all_commands_with_aliases,
    complete_command_names,
    complete_languages,
    complete_hostnames,
    complete_ports,
    get_completion_script,
    install_completion
)


def demo_command_aliases():
    """Demonstrate command aliases functionality."""
    print("Command Aliases Demo")
    print("=" * 50)
    
    print("\nSupported command aliases:")
    for alias, command in COMMAND_ALIASES.items():
        print(f"  - {alias} -> {command}")
    
    print(f"\nAll available commands (including aliases):")
    commands = get_all_commands_with_aliases()
    for i, cmd in enumerate(commands, 1):
        if cmd in COMMAND_ALIASES:
            alias_of = COMMAND_ALIASES[cmd]
            print(f"  {i:2d}. {cmd:12} (alias for {alias_of})")
        else:
            print(f"  {i:2d}. {cmd:12} (main command)")
    
    print(f"\nTotal commands available: {len(commands)}")
    print(f"   - Main commands: {len([c for c in commands if c not in COMMAND_ALIASES])}")
    print(f"   - Aliases: {len([c for c in commands if c in COMMAND_ALIASES])}")


def demo_completion_callbacks():
    """Demonstrate completion callback functions."""
    print("\n\nCompletion Callbacks Demo")
    print("=" * 50)
    
    # Command completion demo
    print("\n1. Command name completion:")
    print("   Empty string (show all):", complete_command_names(None, [], ""))
    print("   Partial 's':", complete_command_names(None, [], "s"))
    print("   Partial 'find':", complete_command_names(None, [], "find"))
    
    # Language completion demo
    print("\n2. Language completion:")
    print("   Empty string (show all):", complete_languages(None, [], ""))
    print("   Partial 'p':", complete_languages(None, [], "p"))
    print("   Partial 'java':", complete_languages(None, [], "java"))
    
    # Hostname completion demo
    print("\n3. Hostname completion:")
    print("   Empty string:", complete_hostnames(None, [], ""))
    print("   Partial '127':", complete_hostnames(None, [], "127"))
    print("   Partial 'local':", complete_hostnames(None, [], "local"))
    
    # Port completion demo
    print("\n4. Port completion:")
    print("   Empty string:", complete_ports(None, [], ""))
    print("   Partial '80':", complete_ports(None, [], "80"))
    print("   Partial '5':", complete_ports(None, [], "5"))


def demo_completion_scripts():
    """Demonstrate shell completion scripts."""
    print("\n\nShell Completion Scripts Demo")
    print("=" * 50)
    
    for shell in ["bash", "zsh", "fish"]:
        print(f"\n{shell.upper()} Completion Script:")
        print("-" * 30)
        
        script = get_completion_script(shell)
        print(f"Length: {len(script)} characters")
        
        # Show first few lines
        lines = script.split('\n')[:5]
        for line in lines:
            print(f"  {line}")
        print("  ...")


def demo_supported_languages():
    """Demonstrate supported languages for completion."""
    print("\n\nSupported Languages Demo")
    print("=" * 50)
    
    print(f"\nTotal supported languages: {len(SUPPORTED_LANGUAGES)}")
    
    # Group languages by first letter
    from collections import defaultdict
    by_letter = defaultdict(list)
    for lang in SUPPORTED_LANGUAGES:
        by_letter[lang[0].upper()].append(lang)
    
    for letter in sorted(by_letter.keys()):
        languages = by_letter[letter]
        print(f"  {letter}: {', '.join(languages)}")


def demo_installation_examples():
    """Show completion installation examples."""
    print("\n\nInstallation Examples")
    print("=" * 50)
    
    print("\n1. Automatic installation:")
    print("   coarch completion --shell bash")
    print("   coarch completion --shell zsh")
    print("   coarch completion --shell fish")
    
    print("\n2. Custom shell rc file:")
    print("   coarch completion --shell bash --rc-file ~/.bash_profile")
    print("   coarch completion --shell zsh --rc-file ~/.zshrc")
    
    print("\n3. Show completion script only:")
    print("   coarch completion --shell bash --no-install")
    print("   coarch completion --shell zsh --no-install")
    print("   coarch completion --shell fish --no-install")
    
    print("\n4. Usage examples after installation:")
    print("   coarch <TAB>                    # Show all commands")
    print("   coarch f<TAB>                  # Complete to 'find' (alias for search)")
    print("   coarch search <TAB>             # Show search options")
    print("   coarch index /path/<TAB>        # Complete directory paths")
    print("   coarch search --language p<TAB>  # Complete language names")
    print("   coarch serve --host <TAB>       # Complete hostnames")
    print("   coarch serve --port <TAB>      # Complete port numbers")


def demo_use_cases():
    """Show practical use cases."""
    print("\n\nPractical Use Cases")
    print("=" * 50)
    
    print("\n1. Command Discovery:")
    print("   Instead of remembering 'coarch status', you can use:")
    print("   - coarch stats    (alias for status)")
    print("   - coarch ping     (alias for health)")
    print("   - coarch find     (alias for search)")
    
    print("\n2. Language-Specific Search:")
    print("   coarch search --language python <TAB>")
    print("   # Shows: python, perl, powershell...")
    
    print("\n3. Server Configuration:")
    print("   coarch serve --host <TAB>")
    print("   # Shows: localhost, 127.0.0.1, 0.0.0.0")
    print("   coarch serve --port <TAB>")
    print("   # Shows: 8000, 8080, 3000, 5000, 9000...")
    
    print("\n4. Path Completion:")
    print("   coarch index /usr/<TAB>")
    print("   # Shows: /usr/local, /usr/bin, /usr/include...")
    
    print("\n5. Quick Health Check:")
    print("   coarch ping --port 8<TAB>")
    print("   # Shows: 8000, 8080")


def main():
    """Run all demos."""
    print("CLI-6: Command Aliases and Tab Completion Demo")
    print("=" * 60)
    print("This demo shows the features implemented for CLI-6:")
    print("- Command aliases for improved discoverability")
    print("- Shell completion for multiple shells (bash, zsh, fish)")
    print("- Context-aware completion for arguments")
    print("- Easy installation and configuration")
    
    # Run all demos
    demo_command_aliases()
    demo_supported_languages()
    demo_completion_callbacks()
    demo_completion_scripts()
    demo_installation_examples()
    demo_use_cases()
    
    print("\n\nDemo completed successfully!")
    print("To test the completion in your shell:")
    print("1. Run: python -m cli.main completion --shell bash")
    print("2. Follow the installation instructions")
    print("3. Restart your shell or source the completion file")
    print("4. Try: coarch <TAB> to see all available commands")


if __name__ == "__main__":
    main()