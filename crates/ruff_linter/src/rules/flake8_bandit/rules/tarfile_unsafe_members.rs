use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::{self as ast};
use ruff_python_semantic::Modules;
use ruff_text_size::Ranged;

use crate::Violation;
use crate::checkers::ast::Checker;

/// ## What it does
/// Checks for uses of `tarfile.extractall`.
///
/// ## Why is this bad?
///
/// Extracting archives from untrusted sources without prior inspection is
/// a security risk, as maliciously crafted archives may contain files that
/// will be written outside of the target directory. For example, the archive
/// could include files with absolute paths (e.g., `/etc/passwd`), or relative
/// paths with parent directory references (e.g., `../etc/passwd`).
///
/// On Python 3.12 and later, use `filter='data'` to prevent the most dangerous
/// security issues (see: [PEP 706]). On earlier versions, set the `members`
/// argument to a trusted subset of the archive's members.
///
/// ## Example
/// ```python
/// import tarfile
/// import tempfile
///
/// tar = tarfile.open(filename)
/// tar.extractall(path=tempfile.mkdtemp())
/// tar.close()
/// ```
///
/// ## References
/// - [Common Weakness Enumeration: CWE-22](https://cwe.mitre.org/data/definitions/22.html)
/// - [Python documentation: `TarFile.extractall`](https://docs.python.org/3/library/tarfile.html#tarfile.TarFile.extractall)
/// - [Python documentation: Extraction filters](https://docs.python.org/3/library/tarfile.html#tarfile-extraction-filter)
///
/// [PEP 706]: https://peps.python.org/pep-0706/#backporting-forward-compatibility
#[derive(ViolationMetadata)]
pub(crate) struct TarfileUnsafeMembers;

impl Violation for TarfileUnsafeMembers {
    #[derive_message_formats]
    fn message(&self) -> String {
        "Uses of `tarfile.extractall()`".to_string()
    }
}

/// S202
pub(crate) fn tarfile_unsafe_members(checker: &Checker, call: &ast::ExprCall) {
    if !checker.semantic().seen_module(Modules::TARFILE) {
        return;
    }

    if call
        .func
        .as_attribute_expr()
        .is_none_or(|attr| attr.attr.as_str() != "extractall")
    {
        return;
    }

    if call
        .arguments
        .find_keyword("filter")
        .and_then(|keyword| keyword.value.as_string_literal_expr())
        .is_some_and(|value| matches!(value.value.to_str(), "data" | "tar"))
    {
        return;
    }

    checker.report_diagnostic(TarfileUnsafeMembers, call.func.range());
}
