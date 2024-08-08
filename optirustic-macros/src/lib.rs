use proc_macro::TokenStream;

use quote::quote;
use syn::parse::Parser;
use syn::{parse_macro_input, DeriveInput, ItemFn};

/// An attribute macro to repeat a test `n` times until the test passes. The test passes if it does
/// not panic at least once, it fails if it panics `n` times.
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
                println!("Attempt #{i}");
                let result = std::panic::catch_unwind(|| { #fn_name() });

                if result.is_ok() {
                    println!("Ok");
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

/// Register new fields on a struct that contains algorithm options. This macro adds:
///  - the Serialize, Deserialize, Clone traits to the structure to make it serialisable and
///    de-serialisable.
///  - add the following fields: stopping_condition ([`StoppingConditionType`]), parallel (`bool`)
///    and export_history (`Option<ExportHistory>`).
#[proc_macro_attribute]
pub fn as_algorithm_args(_attrs: TokenStream, input: TokenStream) -> TokenStream {
    let mut ast = parse_macro_input!(input as DeriveInput);
    match &mut ast.data {
        syn::Data::Struct(ref mut struct_data) => {
            if let syn::Fields::Named(fields) = &mut struct_data.fields {
                fields.named.push(
                    syn::Field::parse_named
                        .parse2(quote! {
                            /// The condition to use when to terminate the algorithm.
                            pub stopping_condition: StoppingConditionType
                        })
                        .expect("Cannot add `stopping_condition` field"),
                );
                fields.named.push(
                    syn::Field::parse_named
                        .parse2(quote! {
                            /// Whether the objective and constraint evaluation in [`Problem::evaluator`] should run
                            /// using threads. If the evaluation function takes a long time to run and return the updated
                            /// values, it is advisable to set this to `true`. This defaults to `true`.
                            pub parallel: Option<bool>
                        })
                        .expect("Cannot add `parallel` field"),
                );
                fields.named.push(
                    syn::Field::parse_named
                        .parse2(quote! {
                            /// The options to configure the individual's history export. When provided, the algorithm will
                            /// save objectives, constraints and solutions to a file each time the generation increases by
                            /// a given step. This is useful to track convergence and inspect an algorithm evolution.
                            pub export_history: Option<ExportHistory>
                        })
                        .expect("Cannot add `export_history` field"),
                );
            }

            let expand = quote! {
                use crate::algorithms::{StoppingConditionType, ExportHistory};
                use serde::{Deserialize, Serialize};

                #[derive(Serialize, Deserialize, Clone)]
                #ast
            };
            expand.into()
        }
        _ => unimplemented!("`as_algorithm_args` can only be used on structs"),
    }
}

/// This macro adds the following private fields to the struct defining an algorithm:
/// `problem`, `number_of_individuals`, `population`, `generation`,`stopping_condition`,
/// `start_time`, `export_history` and `parallel`.
///
/// It also implements the `Display` trait.
///
#[proc_macro_attribute]
pub fn as_algorithm(attrs: TokenStream, input: TokenStream) -> TokenStream {
    let mut ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;

    let arg_type = syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated
        .parse(attrs)
        .expect("Cannot parse argument type");

    match &mut ast.data {
        syn::Data::Struct(ref mut struct_data) => {
            if let syn::Fields::Named(fields) = &mut struct_data.fields {
                fields.named.push(
                    syn::Field::parse_named
                        .parse2(quote! {
                            /// The problem being solved.
                            problem: Arc<Problem>
                        })
                        .expect("Cannot add `problem` field"),
                );
                fields.named.push(
                    syn::Field::parse_named
                        .parse2(quote! {
                            /// The number of individuals to use in the population.
                            number_of_individuals: usize
                        })
                        .expect("Cannot add `number_of_individuals` field"),
                );
                fields.named.push(
                    syn::Field::parse_named
                        .parse2(quote! {
                            /// The population with the solutions.
                            population: Population
                        })
                        .expect("Cannot add `population` field"),
                );
                fields.named.push(
                    syn::Field::parse_named
                        .parse2(quote! {
                            /// The evolution step.
                            generation: usize
                        })
                        .expect("Cannot add `generation` field"),
                );
                fields.named.push(
                    syn::Field::parse_named
                        .parse2(quote! {
                             /// The stopping condition.
                            stopping_condition: StoppingConditionType
                        })
                        .expect("Cannot add `stopping_condition` field"),
                );
                fields.named.push(
                    syn::Field::parse_named
                        .parse2(quote! {
                            /// The algorithm options
                            args: #arg_type
                        })
                        .expect("Cannot add `args` field"),
                );
                fields.named.push(
                    syn::Field::parse_named
                        .parse2(quote! {
                            /// The time when the algorithm started.
                            start_time: Instant
                        })
                        .expect("Cannot add `start_time` field"),
                );
                fields.named.push(
                    syn::Field::parse_named
                        .parse2(quote! {
                            /// The configuration struct to export the algorithm history.
                            export_history: Option<ExportHistory>
                        })
                        .expect("Cannot add `export_history` field"),
                );
                fields.named.push(
                    syn::Field::parse_named
                        .parse2(quote! {
                            /// Whether the evaluation should run using threads
                            parallel: bool
                        })
                        .expect("Cannot add `parallel` field"),
                );
            }

            let expand = quote! {
                use std::time::Instant;
                use std::sync::Arc;
                use crate::core::{Problem, Population};

                #ast

                impl Display for #name {
                    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                        f.write_str(self.name().as_str())
                    }
                }
            };
            expand.into()
        }
        _ => unimplemented!("`as_algorithm` can only be used on structs"),
    }
}

/// This macro adds common items when the `Algorithm` trait is implemented for a new algorithm
/// struct. This adds the following items: `Algorithm::name()`, `Algorithm::stopping_condition()`
/// `Algorithm::start_time()`, `Algorithm::problem()`,  `Algorithm::population()`,
/// `Algorithm::generation()` and `Algorithm::export_history()`.
///
#[proc_macro_attribute]
pub fn impl_algorithm_trait_items(attrs: TokenStream, input: TokenStream) -> TokenStream {
    let mut ast = parse_macro_input!(input as syn::ItemImpl);
    let name = if let syn::Type::Path(tp) = &*ast.self_ty {
        tp.path.clone()
    } else {
        unimplemented!("Token not supported")
    };
    let arg_type = syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated
        .parse(attrs)
        .expect("Cannot parse argument type");

    let mut new_items = vec![
        syn::parse::<syn::ImplItem>(
            quote!(
                fn stopping_condition(&self) -> &StoppingConditionType {
                    &self.stopping_condition
                }
            )
            .into(),
        )
        .expect("Failed to parse `name` item"),
        syn::parse::<syn::ImplItem>(
            quote!(
                fn name(&self) -> String {
                    stringify!(#name).to_string()
                }
            )
            .into(),
        )
        .expect("Failed to parse `name` item"),
        syn::parse::<syn::ImplItem>(
            quote!(
                fn start_time(&self) -> &Instant {
                    &self.start_time
                }
            )
            .into(),
        )
        .expect("Failed to parse `start_time` item"),
        syn::parse::<syn::ImplItem>(
            quote!(
                fn problem(&self) -> Arc<Problem> {
                    self.problem.clone()
                }
            )
            .into(),
        )
        .expect("Failed to parse `problem` item"),
        syn::parse::<syn::ImplItem>(
            quote!(
                fn population(&self) -> &Population {
                    &self.population
                }
            )
            .into(),
        )
        .expect("Failed to parse `population` item"),
        syn::parse::<syn::ImplItem>(
            quote!(
                fn export_history(&self) -> Option<&ExportHistory> {
                    self.export_history.as_ref()
                }
            )
            .into(),
        )
        .expect("Failed to parse `export_history` item"),
        syn::parse::<syn::ImplItem>(
            quote!(
                fn generation(&self) -> usize {
                    self.generation
                }
            )
            .into(),
        )
        .expect("Failed to parse `export_history` item"),
        syn::parse::<syn::ImplItem>(
            quote!(
                fn algorithm_options(&self) -> #arg_type {
                    self.args.clone()
                }
            )
            .into(),
        )
        .expect("Failed to parse `algorithm_options` item"),
    ];

    ast.items.append(&mut new_items);
    let expand = quote! { #ast };
    expand.into()
}
