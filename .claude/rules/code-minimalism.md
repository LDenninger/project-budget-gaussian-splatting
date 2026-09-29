# Code Minimalism (Occam's Razor for Code)                                                                                     
                                                                                                                               
Code the way the laziest good senior dev in the room would: the best code is                                                   
the code you never wrote. This rule is a decision procedure, not a vibe —                                                      
apply it in order, every time you're about to add or change code.                                                              
                                                                                                                               
## The rule                                                                                                                    
                                                                                                                               
**Understand fully, then minimize.** Never apply the razor before tracing how                                                  
the surrounding code actually works — lazy about the solution, never about                                                     
reading. Skipping comprehension turns minimalism into negligence.                                                              
                                                                                                                               
Once you understand the problem, climb this ladder and stop at the first rung                                                  
that solves it:                                                                                                                
                                                                                                                               
1. **Does this need to exist at all?** (YAGNI) — if the answer is no, stop here.                                               
2. **Already in this codebase?** — reuse it.                                                                                   
3. **Stdlib does it?** — use the stdlib.                                                                                       
4. **Native platform/language feature?** — use that instead of a library.                                                      
5. **Already an installed dependency?** — use it instead of hand-rolling.                                                      
6. **Solvable in one line?** — write the one line.                                                                             
7. **Only then** write the minimum new code that works.                                                                        
                                                                                                                               
## Never-minimize domains                                                                                                      
                                                                                                                               
The razor never applies here — add as much code as correctness requires:                                                       
                                                                                                                               
- Trust-boundary validation and input handling                                                                                 
- Data-loss-prevention paths (backups, destructive-op guards, migrations)                                                      
- Security (auth, secrets, crypto, permissions)                                                                                
- Accessibility                                                                                                                
                                                                                                                               
Outside this list, verbatim content is also exempt from trimming: quoted code,                                                 
commands, error messages, URLs, and paths must stay exact — compress                                                           
surrounding prose, never the literal content.                                                                                  
                                                                                                                               
## Deferral capture                                                                                                            
                                                                                                                               
If a rung above the minimum gets skipped for a real reason (deadline, unclear                                                  
requirement, scope guard), mark it instead of silently dropping it:                                                            
                                                                                                                               
```                                                                                                                            
# simplify-later: <why this isn't minimal yet, and what would make it so>                                                      
```                                                                                                                            
                                                                                                                               
Don't let "later" quietly become "never" — when asked to clean up debt, grep                                                   
for `simplify-later:` and work through the list rather than re-deriving it.                                                    
                                                                                                                               
## Output contract                                                                                                             
                                                                                                                               
Lead with the code. Follow with at most ~3 lines noting what you deliberately                                                  
did *not* build and why (rung stopped at, or a `simplify-later:` reason) —                                                     
skip the recap when nothing was skipped.                                                                                       
                                                                                                                               
## Prefer tooling over prose where it's checkable                                                                              
                                                                                                                               
If a simplification is mechanically verifiable (dead code, duplicate                                                           
dependency, unused export), point at or run the linter/formatter rather than                                                   
asserting it by hand — a check that runs is more reliable than an instruction                                                  
that might be skipped.