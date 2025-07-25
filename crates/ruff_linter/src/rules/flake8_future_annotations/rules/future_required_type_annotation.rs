use std::fmt;

use ruff_macros::{ViolationMetadata, derive_message_formats};
use ruff_python_ast::Expr;
use ruff_text_size::Ranged;

use crate::checkers::ast::Checker;
use crate::{AlwaysFixableViolation, Fix};

/// ## What it does
/// Checks for uses of PEP 585- and PEP 604-style type annotations in Python
/// modules that lack the required `from __future__ import annotations` import
/// for compatibility with older Python versions.
///
/// ## Why is this bad?
/// Using PEP 585 and PEP 604 style annotations without a `from __future__ import
/// annotations` import will cause runtime errors on Python versions prior to
/// 3.9 and 3.10, respectively.
///
/// By adding the `__future__` import, the interpreter will no longer interpret
/// annotations at evaluation time, making the code compatible with both past
/// and future Python versions.
///
/// This rule respects the [`target-version`] setting. For example, if your
/// project targets Python 3.10 and above, adding `from __future__ import annotations`
/// does not impact your ability to leverage PEP 604-style unions (e.g., to
/// convert `Optional[str]` to `str | None`). As such, this rule will only
/// flag such usages if your project targets Python 3.9 or below.
///
/// ## Example
///
/// ```python
/// def func(obj: dict[str, int | None]) -> None: ...
/// ```
///
/// Use instead:
///
/// ```python
/// from __future__ import annotations
///
///
/// def func(obj: dict[str, int | None]) -> None: ...
/// ```
///
/// ## Fix safety
/// This rule's fix is marked as unsafe, as adding `from __future__ import annotations`
/// may change the semantics of the program.
///
/// ## Options
/// - `target-version`
#[derive(ViolationMetadata)]
pub(crate) struct FutureRequiredTypeAnnotation {
    reason: Reason,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Reason {
    /// The type annotation is written in PEP 585 style (e.g., `list[int]`).
    PEP585,
    /// The type annotation is written in PEP 604 style (e.g., `int | None`).
    PEP604,
}

impl fmt::Display for Reason {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Reason::PEP585 => fmt.write_str("PEP 585 collection"),
            Reason::PEP604 => fmt.write_str("PEP 604 union"),
        }
    }
}

impl AlwaysFixableViolation for FutureRequiredTypeAnnotation {
    #[derive_message_formats]
    fn message(&self) -> String {
        let FutureRequiredTypeAnnotation { reason } = self;
        format!("Missing `from __future__ import annotations`, but uses {reason}")
    }

    fn fix_title(&self) -> String {
        "Add `from __future__ import annotations`".to_string()
    }
}

/// FA102
pub(crate) fn future_required_type_annotation(checker: &Checker, expr: &Expr, reason: Reason) {
    checker
        .report_diagnostic(FutureRequiredTypeAnnotation { reason }, expr.range())
        .set_fix(Fix::unsafe_edit(checker.importer().add_future_import()));
}
