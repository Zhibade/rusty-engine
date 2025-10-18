//! Derive macros implementation

use proc_macro::{self, TokenStream};
use quote::quote;
use syn::{DeriveInput, parse_macro_input};

/// Derive macro implementation for Component trait
#[proc_macro_derive(Component)]
pub fn derive_component(input: TokenStream) -> TokenStream
{
    let DeriveInput { ident, .. } = parse_macro_input!(input);
    let output = quote! {
        impl Component for #ident
        {
            fn as_any(&self) -> &dyn Any
            {
                self as &dyn Any
            }

            fn as_any_mut(&mut self) -> &mut dyn Any
            {
                self as &mut dyn Any
            }
        }
    };

    output.into()
}
