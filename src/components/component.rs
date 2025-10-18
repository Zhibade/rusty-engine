//! Component trait definition

use std::any::Any;

/// Identifies a struct as a component
///
/// All components have to implement this trait to be supported by [World]
pub trait Component: Any
{
    /// Return component as Any trait (immutable)
    fn as_any(&self) -> &dyn Any;

    /// Return component as Any trait (mutable)
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
