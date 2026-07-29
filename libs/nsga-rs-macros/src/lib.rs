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
///  - the `Serialize`, `Deserialize`, `Clone` traits to the structure to make it
///     serialisable and de-serialisable.
///  - the `#[cfg_attr(feature = "python", pyclass(get_all, from_py_object))]` macro for
///    Python feature.
///  - add the following fields:
///      - crossover_operator_options ([`Option<SimulatedBinaryCrossoverArgs>`])
///      - mutation_operator_options ([`Option<PolynomialMutationArgs>`])
///      - stopping_condition ([`StoppingCondition`])
///      - threads ([`NumThreads`])
///      - export_history ([`ExportHistory`])
///      - resume_from_file ([`Option<PathBuf>`])
///      - seed ([`Option<u64>`])
#[proc_macro_attribute]
pub fn algorithm_args(_attrs: TokenStream, input: TokenStream) -> TokenStream {
    let mut ast = parse_macro_input!(input as DeriveInput);
    match &mut ast.data {
        syn::Data::Struct(ref mut struct_data) => {
            if let syn::Fields::Named(fields) = &mut struct_data.fields {
                fields.named.push(
                    syn::Field::parse_named
                        .parse2(quote! {
                            /// The options of the Simulated Binary Crossover (SBX) operator. This operator is used to
                            /// generate new children by recombining the variables of parent solutions. This defaults to
                            /// [`SimulatedBinaryCrossoverArgs::default()`].
                            pub crossover_operator_options: Option<SimulatedBinaryCrossoverArgs>
                        })
                        .expect("Cannot add `crossover_operator_options` field"),
                );
                fields.named.push(
                    syn::Field::parse_named
                        .parse2(quote! {
                        /// The options to Polynomial Mutation (PM) operator used to mutate the variables of an
                        /// individual. This defaults to [`PolynomialMutationArgs::default()`],
                        /// with a distribution index or index parameter of `20` and variable probability equal `1`
                        /// divided by the number of real variables in the problem (i.e., each variable will have the
                        /// same probability of being mutated).
                            pub mutation_operator_options: Option<PolynomialMutationArgs>
                        })
                        .expect("Cannot add `mutation_operator_options` field"),
                );
                fields.named.push(
                    syn::Field::parse_named
                        .parse2(quote! {
                            /// The condition to use when to terminate the algorithm.
                            pub stopping_condition: StoppingCondition
                        })
                        .expect("Cannot add `stopping_condition` field"),
                );
                fields.named.push(
                    syn::Field::parse_named
                        .parse2(quote! {
                            /// Instead of initialising the population with random variables, see the initial population
                            /// with  the variable values from a JSON files exported with this tool. This option lets you
                            /// restart the evolution from a previous generation; you can use any history file (exported
                            /// when the field `export_history`) or the file exported when the stopping condition was reached.
                            pub resume_from_file: Option<PathBuf>
                        })
                        .expect("Cannot add `resume_from_file` field"),
                );
                fields.named.push(
                    syn::Field::parse_named
                        .parse2(quote! {
                            /// The seed used in the random number generator (RNG). You can specify a seed in case you want
                            /// to try to reproduce results. NSGA2 is a stochastic algorithm that relies on a RNG at
                            /// different steps (when population is initially generated, during selection, crossover and
                            /// mutation) and, as such, may lead to slightly different solutions. The seed is randomly
                            /// picked if this is `None`.
                            pub seed: Option<u64>
                        })
                        .expect("Cannot add `seed` field"),
                );
                fields.named.push(
                    syn::Field::parse_named
                        .parse2(quote! {
                            /// The number of threads to use to parallel evaluate the objectives and constraints
                            /// in [`Problem::evaluator`]. If the evaluation function takes a long time to run,
                            /// it is advisable to set this option.
                            pub threads: NumThreads
                        })
                        .expect("Cannot add `threads` field"),
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
                use crate::algorithms::{StoppingCondition, ExportHistory};
                use serde::{Deserialize, Serialize};

                #[derive(Serialize, Deserialize, Clone)]
                #[cfg_attr(feature = "python", pyclass(get_all, from_py_object))]
                #ast
            };
            expand.into()
        }
        _ => unimplemented!("`as_algorithm_args` can only be used on structs"),
    }
}

/// This macro adds the following private fields to the struct defining an algorithm:
/// `problem`, `number_of_individuals`, `population`, `generation`,`stopping_condition`,
/// `number_of_function_evaluations`, `start_time`, `export_history` and `threads`.
///
/// It also implements the `Display` trait.
///
#[proc_macro_attribute]
pub fn algorithm(attrs: TokenStream, input: TokenStream) -> TokenStream {
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
                            generation: u32
                        })
                        .expect("Cannot add `generation` field"),
                );
                fields.named.push(
                    syn::Field::parse_named
                        .parse2(quote! {
                            /// The number of function evaluations.
                            nfe: u32
                        })
                        .expect("Cannot add `nfe` field"),
                );
                fields.named.push(
                    syn::Field::parse_named
                        .parse2(quote! {
                             /// The stopping condition.
                            stopping_condition: StoppingCondition
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
                            /// The thread pool.
                            thread_pool: Option<ThreadPool>
                        })
                        .expect("Cannot add `thread_pool` field"),
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
/// `Algorithm::generation()`, `Algorithm::number_of_function_evaluations()`,
/// `Algorithm::build_thread_pool()` and `Algorithm::export_history()`.
///
#[proc_macro_attribute]
pub fn algorithm_trait_items(attrs: TokenStream, input: TokenStream) -> TokenStream {
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
                fn stopping_condition(&self) -> &StoppingCondition {
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
                fn generation(&self) -> u32 {
                    self.generation
                }
            )
            .into(),
        )
        .expect("Failed to parse `generation` item"),
        syn::parse::<syn::ImplItem>(
            quote!(
                fn generation_as_ref(&self) -> &u32 {
                    &self.generation
                }
            )
            .into(),
        )
        .expect("Failed to parse `generation_as_ref` item"),
        syn::parse::<syn::ImplItem>(
            quote!(
                fn number_of_function_evaluations(&self) -> u32 {
                    self.nfe
                }
            )
            .into(),
        )
        .expect("Failed to parse `number_of_function_evaluations` item"),
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
