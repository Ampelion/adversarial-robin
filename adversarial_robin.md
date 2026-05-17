# The Adversarial Robin

### In pursuit of an AI scientist

Josie Babin · May 2026

AI models have breadth of factual recall beyond that of any human, there is no such thing (yet) as an AI scientist. An AI's factual repertoire and the inferences made therefrom derive from what scientists (or any other human) wrote down. A peer-reviewed article is a hard-thought, iterated communication of their activities streamlined to accentuate the contribution being made with as much noise stripped out as possible. The AI is tasked with doing science by learning from the intellectual residue of cognitive and physical action undertaken by imperfect beings whose own minds sometimes fail to recognize how they know things.

A human Principal Investigator directing their lab does not build a career on recall of work that came before. Recall is necessary but insufficient; knowing the literature is mandatory but choosing what question to ask in light of it plus new observations is an intercalating layer essential to the process. Hypotheses get rejected with a wrinkled lip, reasoning gets bashed and honed in lab meetings, anomalous readings from an instrument get the side-eye before any formal write-up. Science as a process, a human endeavor, is replete with messy diversions throughout and its mostly polished output makes it to the training data.

Reading articles does not make a person a scientist; it provides context and a sense of place for ideas. Internal motivation and curiosity, combined with reasoning and work, make the contribution. Throwing more compute at an AI will not replace the human endeavor, but there might be a way to approximate the human tacit layer.

Polanyi: we can know more than we can tell. The knowledge lives in the practice. It is encoded in behavior and output, not in self-report.

The tacit dimension has two layers worth distinguishing. The object-level layer is pre-verbal and embodied: the pathologist's eye, the cyclist's body sense, the bench scientist's hands. This layer cannot translate to text. The meta-cognitive layer sits one step up: following a question that feels right, recognizing weirdness before you can say why, critical thinking turned inward. This layer is partially articulable. It gets stripped from published outputs but remains in conversation.

LLMs are conversational and helpful. A user reasons through a thought experiment, providing a thorough explanation, the hedges and caveats, recognized biases and associated minutia, because the output quality hinges on the reasoning input quality. People explain things to an LLM they would streamline in a paper.

The prompt record is a potentially rich site of expression for the meta-cognitive tacit layer. The embodied layer is irrelevant to an AI scientist anyway, or has to be substituted by direct sensor data. The meta-cognitive layer is the part an AI scientist would need to graduate from post-doc to PI.

Retaining chat prompt elaborations over time and applying them generatively to inference could get AI closer to behaving scientifically, or at least more like that particular scientist.

The retained verbiage is not tacit knowledge itself. It must be processed and picked through, then inverted and run forward to enrich responses to forthcoming prompts. The test of utility is in predictive competence, whether the next prompt, or scientific step, is inferable.

---

This is not new. Inversion is the standard tool for myriad data-sparse problems. Before DNA sequencing was trivial, molecular phylogenies were inferred from snippets run through analytical parkour that strained the compute of the day: sequences are observable outputs, evolutionary history is the latent generative structure, inversion is the move from sequence to tree.

Current LLM prompts are used for conditioning — input-output pairs to learn response patterns from. Can we also use them as behavioral traces to invert from? The data that would support inversion is there. The method is not.

A further lesson from sequencing is cautionary: Those early snippets were sometimes viewed as ground truth for evolutionary history, and deference to DNA over other data was default.

This did not hold up under scrutiny, of course, but the mistake was easy to make. DNA is a proxy for evolutionary process, not the process itself — the inversion from sequence to tree depends on the model assumed, and the model assumed was rarely as nuanced as the confidence in the conclusion. DNA is not a monolith and it cannot be taken as a stepwise record of evolution. Meta-cognitive traces will surely have their own nuance that demands interrogation: differences in method and thinking styles among scientists, domain differences, problem tractability.

This is the cautionary principle for naive inversion of prompt content. Without rigorous validation, our inversion will be locally consistent, demographically plausible, and confident. It will reproduce the biases of the recovery method rather than the structure of the practitioner. Kosinski et al. 2013 demonstrated that OCEAN traits can be recovered from Facebook likes. That is the prior art, a personality quiz with extra steps.

---

Not every prompt is rich in process residue. Imagine all prompts get tossed into a compost bin, with worms throughout, processing every scrap that appears. What if we empowered the worms to wriggle through the compost at different rates (temperatures) from different starting points? The worms can explore the compost's features with more or less interest in the component parts like MCMC in a Bayesian interrogation of a posterior probability distribution.

Then along comes a hungry robin, intent on jackknifing the whole production from out of nowhere. It plucks out worms at random, depriving the compost system of a small bit of functionality. Different worms lost, different shape of compost, what is the compost shape after the robin's visit? This is the falsification engine attacking the production engine. They are separate and the one hones the other: How robust are the shapes in the compost to the plucking out of its worms?

---

Right now, the system that we're shaping to become an AI scientist is closer to a scientific cargo cult. It has learned the products of critical thinking: methodology sections, limitations paragraphs, amiable steel-manning of the objections, all without the practice that produced them, because that practice happens in human space before the writing.

It can produce a limitations paragraph; it did not generate the limitation by trying to break its own claim. When it looks like it is thinking critically it was prompted to do so, or it is naming objections prescribed in the training data. It is fluent. The understanding is not there. No plane lands on a runway of mimicry.

Robinson Crusoe's crude home stands because he understands why it stands. Even with better materials, or more compute, the cargo cult still builds a sham runway that never summons a plane.

Tacit knowledge cannot be transferred, but we'll be closer to AI science if enough of its underlying shape can be recovered by inversion and the robin's attack.

---

## Works held in mind while writing

Defoe, D. (1719). *Robinson Crusoe*. London: W. Taylor.

Feynman, R. P. (1974). Cargo cult science. *Engineering and Science*, 37(7), 10–13. Caltech commencement address; reprinted in *Surely You're Joking, Mr. Feynman!* (W. W. Norton, 1985).

Hull, D. L. (1988). *Science as a Process: An Evolutionary Account of the Social and Conceptual Development of Science*. University of Chicago Press.

Kosinski, M., Stillwell, D., & Graepel, T. (2013). Private traits and attributes are predictable from digital records of human behavior. *Proceedings of the National Academy of Sciences*, 110(15), 5802–5805. https://doi.org/10.1073/pnas.1218772110

Polanyi, M. (1966). *The Tacit Dimension*. Doubleday. (Reprint: University of Chicago Press, 2009.)
