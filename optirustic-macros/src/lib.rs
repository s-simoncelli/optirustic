use proc_macro::TokenStream;

use quote::quote;
use syn::{parse_macro_input, ItemFn};

/// An attribute macro to repeat a test `n` times until the test passes. The test passes if it does
/// not panic once, it fails if it panics `n` times.
#[proc_macro_attribute]
pub fn test_with_retries(attrs: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let tries = attrs
        .to_string()
        .parse::<u8>()
        .expect("Attr must be an int");

    let expanded = quote! {
        #[test]
        fn #fn_name() {
            #input_fn
            for i in 1..=#tries {
                let result = std::panic::catch_unwind(|| { #fn_name() });

                if result.is_ok() {
                    return;
                }

                if i == #tries {
                    std::panic::resume_unwind(result.unwrap_err());
                }
            };
        }
    };
    expanded.into()
}
